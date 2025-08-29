#!/usr/bin/env python3
"""
copilot_combined_gui_ss.py

This script combines both real-time order book and trades data collection and display.
It runs two collectors in the background:
  • OrderBookCollector – uses asyncio & websockets to obtain a live order book 
    (aggregated into $10 bins over ±$200).
  • CopilotTradesCollector – uses websocket-client to receive live aggregated trades data.
    Instead of handling every trade, it aggregates trades over a snapshot interval and
    produces a “bar” summary that is then displayed in the GUI.
    
The GUI (built with Tkinter) shows three tabs:
  • "Order Book": displays aggregated ask levels, the mid price (integrated into the pane),
    and bid levels.
  • "Trades": displays live trade statistics and a table of the aggregated “bar” data.
  • "Signals": displays live calculated signals based on order book metrics:
       – Trade Imbalance Ratio (TIR): EMA of (Buy Volume – Sell Volume)/(Total Volume), then scaled by x100.
       – Weighted Bid-Ask Imbalance Ratio (WBAIR): EMA of a weighted liquidity imbalance based on a Gaussian weighting,
         scaled by x(-1000).
       – Sustained Liquidity Flow Indicator (SLFI): EMA of the near-market liquidity difference computed separately 
         for bids and asks using an adaptive near-zone, compared as EMA(bid_near) – EMA(ask_near).
       – Cumulative Trade Volume Delta (CTVD): Cumulative net trade volume delta (Buy Volume – Sell Volume) over a rolling window.
    
A top control panel shows a connection status indicator, a Start/Stop button,
a Snapshot Interval entry (in seconds) that sets the snapshot interval for the collectors and the GUI refresh rate,
and a Rolling Window (# snapshots) entry for the rolling history (and EMA smoothing) used in signal calculations.
"""

# version 33a

import socket
import asyncio
import websockets
import aiohttp
import json
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import websocket  # for trades collector
import queue
import math
from collections import deque
import requests
import sys
import numpy as np
from scipy.signal import savgol_filter
from tkinter.scrolledtext import ScrolledText



class TextRedirector:
    """Redirect writes to a Tk Text widget via tags."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag    = tag
    def write(self, message):
        # enable, insert, disable, and autoscroll
        self.widget.configure(state="normal")
        self.widget.insert("end", message, (self.tag,))
        self.widget.configure(state="disabled")
        self.widget.see("end")
    def flush(self):
        pass


# Central configuration for metric constants
class Config:
    # CTVD
    CTVD_EMA_WEIGHT         = 0.8      # effective alpha multiplier for CTVD EMA
    CTVD_NORM_HISTORY_MAX   = 50       # maxlen for CTVD norm deque

    # WBAIR
    WBAIR_SIGMA             = 85      # Gaussian sigma (increase for broader weighting)
    WBAIR_EMA_WEIGHT        = 0.90      # effective alpha multiplier for WBAIR EMA (smaller for smoother response)
    WBAIR_SCALE_FACTOR      = 1000     # raw scale factor before normalize

    # SLFI
    SLFI_LAMBDA             = 10       # multiplier for raw SLFI and mid shift
    SLFI_NEAR_BINS          = 20       # number of bins for near-zone
    SLFI_NORM_HISTORY_MAX   = 50       # maxlen for SLFI norm deque

    # arctan normalization
    # sensitivity constant for arctan_norm, lower K → more aggressive, jumpy, and quickly clipped to ±100, higher K → smoother, more linear, less extreme scaling
    ARCTAN_K                = 1.5      

    # Add‐Imbalance tuning
    ADD_IMB_USE_RELATIVE    = False     # if True, use relative‐diff; else absolute‐diff True False
    ADD_IMB_REL_THRESHOLD   = 1e-3     # min avg bin volume to include in relative‐diff
    ADD_IMB_NEAR_BINS       = 20        # number of bins on each side of mid to consider 10
    ADD_IMB_SIGMA_FACTOR    = 0.7      # sigma = NEAR_BINS * SIGMA_FACTOR 1.0
    # Add-Imbalance smoothing
    ADD_IMB_FILTER_WINDOW   = 5         # Savitzky–Golay window (must be odd)

    # TIR Savitzky–Golay smoothing parameters SG parameter
    TIR_SG_WINDOW           = 5         # window length for SG filter (must be odd)
    TIR_SG_POLYORDER        = 3         # polynomial order for SG filter (≤ window-1)

    # Price Momentum tuning
    PRICEMOM_DAMPING        = 0.75     # fraction of raw delta_price to accumulate (0–1)
    PRICEMOM_ARCTAN_K       = 2.0     # higher K → smaller mapped swings for PriceMOM


print(sys.executable)


# --- Metrics Helpers ---
def sma(data):
    return sum(data)/len(data) if data else 0.0


def ema(new, prev, alpha):
    return alpha*new + (1-alpha)*prev


def zscore(val, history):
    if not history:
        return 0.0
    mean = sum(history)/len(history)
    var  = sum((x-mean)**2 for x in history)/len(history)
    std  = math.sqrt(var) if var>0 else 1e-6
    return (val-mean)/std


def arctan_norm(z, k=2.0):
    return 100 * math.atan(z/k) / (math.pi/2)


def compute_tir(trade_bars, sma_window, ema_window, tir_hist, prev_ema, comb_hist):
    # raw TIR
    if not trade_bars:
        curr = 0.0
    else:
        b = trade_bars[-1]
        total = b.get("total_volume", 0.0)
        curr = ((b.get("buy_volume",0)-b.get("sell_volume",0))/total) if total else 0.0
    # SMA over last sma_window bars
    tir_hist.append(curr)
    s = sma(tir_hist)
    # EMA over ema_window bars
    alpha = 2.0/(ema_window+1)
    base  = curr if prev_ema is None else prev_ema
    e     = ema(curr, base, alpha)
    # Combine then filter
    comb = 0.4 * s + 0.6 * e
    comb_hist.append(comb)
    # apply Savitzky–Golay using our manual constants
    if len(comb_hist) >= Config.TIR_SG_WINDOW:
        wl = Config.TIR_SG_WINDOW
        po = Config.TIR_SG_POLYORDER
        filt = savgol_filter(np.array(comb_hist), wl, po)[-1]
    else:
        filt = comb
    return filt * 100, e


def compute_ctvd(trade_bars, rolling_window, ctvd_hist, prev_ema, norm_hist):
    # raw CTVD: continuous cumulative net volume
    if trade_bars:
        last_bar = trade_bars[-1]
        delta = last_bar.get("buy_volume", 0) - last_bar.get("sell_volume", 0)
    else:
        delta = 0.0
    prev_raw = getattr(compute_ctvd, "cumulative_raw", 0.0)
    cumulative_raw = prev_raw + delta
    setattr(compute_ctvd, "cumulative_raw", cumulative_raw)
    raw = cumulative_raw
    # SMA
    ctvd_hist.append(raw)
    s = sma(ctvd_hist)
    # EMA (70%)
    alpha = Config.CTVD_EMA_WEIGHT * (2.0/(rolling_window+1))
    base  = raw if prev_ema is None else prev_ema
    e     = ema(raw, base, alpha)
    # Combine & normalize
    comb = 0.4 * s + 0.6 * e
    norm_hist.append(comb)
    z = zscore(comb, norm_hist)
    return arctan_norm(z, Config.ARCTAN_K), e


def compute_wbair(snapshot, rolling_window, hist, prev_ema, dyn_hist, sigma=Config.WBAIR_SIGMA):
    if not snapshot or "bins" not in snapshot:
        curr = 0.0
    else:
        mid = snapshot["last_price"]
        num=den=0.0
        for b in snapshot["bins"]:
            low,high = b["bin_range"]
            ctr = (low+high)/2
            w   = math.exp(-((mid-ctr)**2)/(2*sigma**2))
            bid,ask = b.get("bid_qty",0), b.get("ask_qty",0)
            num+= (bid-ask)*w; den+= (bid+ask)*w
        curr = (num/den) if den else 0.0
    hist.append(curr)
    s = sma(hist)
    alpha = Config.WBAIR_EMA_WEIGHT * (2.0/(rolling_window+1))
    base  = curr if prev_ema is None else prev_ema
    e     = ema(curr, base, alpha)
    comb  = 0.3 * s + 0.7 * e
    raw   = comb * Config.WBAIR_SCALE_FACTOR
    dyn_hist.append(raw)
    z = zscore(raw, dyn_hist)
    return arctan_norm(z, Config.ARCTAN_K), e

def compute_liquidity_weighted_mid(snapshot):
    bins = snapshot.get("bins",[])
    bw=bw_tot=aw=aw_tot=0.0
    for b in bins:
        low,high = b["bin_range"]
        mid=(low+high)/2
        bid,ask = b.get("bid_qty",0), b.get("ask_qty",0)
        bw+=bid*mid; bw_tot+=bid
        aw+=ask*mid; aw_tot+=ask
    lwb = (bw/bw_tot) if bw_tot else None
    lwa = (aw/aw_tot) if aw_tot else None
    if lwb is not None and lwa is not None:
        return (lwb+lwa)/2
    return lwb or lwa

def compute_slfi(snapshot, rolling_window,
                 hist, e_bid, e_ask, e_filt, norm_hist,
                 lambda_factor=Config.SLFI_LAMBDA,
                 near_bins=Config.SLFI_NEAR_BINS):
    alpha = 2.0/(rolling_window+1)
    if snapshot and "bins" in snapshot:
        bins      = snapshot["bins"]
        mid       = snapshot["last_price"]
        lw_mid    = compute_liquidity_weighted_mid(snapshot) or 0.0
        mid_shift = lw_mid-mid
        mid_idx   = len(bins)//2
        b_zone    = bins[max(0,mid_idx-near_bins):mid_idx]
        a_zone    = bins[mid_idx:mid_idx+near_bins]
        bid_near  = sum(b["bid_qty"] for b in b_zone)
        ask_near  = sum(b["ask_qty"] for b in a_zone)
    else:
        bid_near=ask_near=mid_shift=0.0
    new_bid = bid_near if e_bid is None else ema(bid_near, e_bid, alpha)
    new_ask = ask_near if e_ask is None else ema(ask_near, e_ask, alpha)
    raw     = (new_bid-new_ask)*lambda_factor
    hist.append(raw)
    s       = sma(hist)
    eff_a   = 0.7*alpha
    new_filt= raw if e_filt is None else ema(raw, e_filt, eff_a)
    comb    = (s + new_filt) / 2 + lambda_factor * mid_shift
    norm_hist.append(comb)
    z       = zscore(comb, norm_hist)
    return arctan_norm(z, Config.ARCTAN_K), new_bid, new_ask, new_filt




#########################################
# TCP Metrics Streamer
#########################################
class MetricsTCPStreamer(threading.Thread):
    """
    This class implements a simple TCP server that listens for client connections
    and can broadcast JSON‐encoded metrics data. Each client receives one JSON string
    per broadcast (terminated by a newline).
    """
    def __init__(self, host="127.0.0.1", port=9999):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.clients = []


        # Lock to protect concurrent access to self.clients
        self._clients_lock = threading.Lock()
        self.running = True
        self.server_socket = None

    def run(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)
            print(f"Metrics TCP Streamer listening on {self.host}:{self.port}")
            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    # ensure sendall won't block indefinitely
                    conn.settimeout(0.5)
                    print("Client connected:", addr)

                    # thread‐safe append
                    with self._clients_lock:
                        self.clients.append(conn)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print("Error accepting connection:", e)
                    break
        except Exception as e:
            print(f"[MetricsTCPStreamer] run loop unexpected error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()


    def broadcast(self, data):
        """
        Converts the data dictionary to JSON (adding a newline) and sends it 
        to each connected client. Removes clients that have disconnected.
        """
        msg = json.dumps(data) + "\n"

        # snapshot the client list under lock
        with self._clients_lock:
            conns = list(self.clients)
        for conn in conns:
            try:
                conn.sendall(msg.encode("utf-8"))
            except Exception as e:
                print("Removing client due to error:", e)
                with self._clients_lock:
                    if conn in self.clients:
                        self.clients.remove(conn)

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        # Close all client connections
        for conn in self.clients:
            try:
                conn.close()
            except Exception:
                pass



#########################################
# Order Book Collector (Asynchronous)
#########################################

class OrderBookCollector:
    def __init__(self, symbol="BTCUSDT", snapshot_interval=1.0):
        self.symbol = symbol.upper()
        self.snapshot_interval = snapshot_interval  # seconds between snapshots
        self.depth_url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit=1000"
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@depth@100ms"
        self.bids = {}
        self.asks = {}
        self.last_update_id = 0
        self.ws = None
        self.is_running = False
        self.latest_snapshot = None
        self.sync_base = None  # set externally by GUI
        # Heartbeat support for WS stalls
        self._last_msg_time   = time.time()   # seed immediately
        self._heartbeat_task  = None
        self.HEARTBEAT_TIMEOUT = 5.0  # seconds without messages triggers reconnect

    async def fetch_snapshot(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(self.depth_url) as resp:
                snapshot = await resp.json()
                self.last_update_id = snapshot['lastUpdateId']
                self.bids = {float(p): float(q) for p, q in snapshot['bids']}
                self.asks = {float(p): float(q) for p, q in snapshot['asks']}
                print(f"[OrderBook] Snapshot initialized. Last Update ID: {self.last_update_id}")

    def apply_update(self, data):
        for price_str, qty_str in data.get("b", []):
            price, qty = float(price_str), float(qty_str)
            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty
        for price_str, qty_str in data.get("a", []):
            price, qty = float(price_str), float(qty_str)
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

    def compute_last_price(self):
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        elif best_bid is not None:
            return best_bid
        elif best_ask is not None:
            return best_ask
        else:
            return None

    def aggregate_bins(self, last_price, bin_width=10, range_width=300, bids=None, asks=None):
         
        #  Builds price‐range bins around last_price and sums bid/ask quantities.
         
        bids_dict = bids if bids is not None else self.bids
        asks_dict = asks if asks is not None else self.asks
        # lower_bound = last_price - range_width
        # upper_bound = last_price + range_width
        # snap center to nearest bin to avoid tiny shifts every tick
        center = bin_width * round(last_price/bin_width)
        lower_bound = center - range_width
        upper_bound = center + range_width
        num_bins = int((upper_bound - lower_bound) / bin_width)
        bins = []
        for i in range(num_bins):
            bin_start = lower_bound + i * bin_width
            bin_end = bin_start + bin_width
            bins.append({
                "bin_range": [bin_start, bin_end],
                "bid_qty": 0.0,
                "ask_qty": 0.0
            })
        for price, qty in bids_dict.items():
            if lower_bound <= price < upper_bound:
                idx = int((price - lower_bound) // bin_width)
                if 0 <= idx < len(bins):
                    bins[idx]["bid_qty"] += qty
        for price, qty in asks_dict.items():
            if lower_bound <= price < upper_bound:
                idx = int((price - lower_bound) // bin_width)
                if 0 <= idx < len(bins):
                    bins[idx]["ask_qty"] += qty
        return bins


    async def snapshot_task(self):
        while self.is_running:
            await asyncio.sleep(self.snapshot_interval)
            last_price = self.compute_last_price()
            if last_price is None:
                print("[OrderBook] No valid last price computed in snapshot_task")
                continue
            # aggregate and capture fixed bin‐center
            bin_width = 10
            bins      = self.aggregate_bins(last_price, bin_width=bin_width, range_width=320)
            center    = bin_width * round(last_price/bin_width)
            snapshot_time = datetime.now().isoformat()
            # include raw orderbook for precise re-binning later
            snapshot_obj = {
                "timestamp": snapshot_time,
                "last_price": last_price,
                "bins": bins,
                "bin_center": center,
                "raw_bids": dict(self.bids),
                "raw_asks": dict(self.asks),
            }
            self.latest_snapshot = snapshot_obj
            # push raw snapshot to signal pipeline
            if hasattr(self, "output_queue"):
                self.output_queue.put(snapshot_obj)
            # print(f"[OrderBook] Snapshot taken at {snapshot_time} with last price {last_price}")

    async def stream_and_collect(self):
        """
        Connect + consume in a reconnect loop with exponential backoff.
        """
        try:
            backoff = 1
            self.is_running = True

            while self.is_running:
                snapshot_future = heartbeat_task = None
                try:
                    # Start heartbeat monitor _before_ REST, so we catch REST stalls
                    if self._heartbeat_task:
                        self._heartbeat_task.cancel()
                    self._heartbeat_task = asyncio.create_task(self._heartbeat())

                    # REST bootstrap with timeout = HEARTBEAT_TIMEOUT
                    self._last_msg_time = time.time()
                    await asyncio.wait_for(
                        self.fetch_snapshot(),
                        timeout=self.HEARTBEAT_TIMEOUT
                    )

                    # open WS
                    self.ws = await websockets.connect(
                        self.ws_url,
                        ping_interval=self.HEARTBEAT_TIMEOUT/2,   # send pings
                        ping_timeout=self.HEARTBEAT_TIMEOUT      # await pongs
                    )

                    print("[OrderBook] WebSocket connected.")

                    # start heartbeat + snapshots
                    heartbeat_task  = asyncio.create_task(self._heartbeat())
                    snapshot_future = asyncio.create_task(self.snapshot_task())

                    # consume updates
                    async for msg in self.ws:
                        self._last_msg_time = time.time()
                        data = json.loads(msg)
                        if data.get('u', 0) <= self.last_update_id:
                            continue
                        self.apply_update(data)
                        self.last_update_id = data['u']

                except Exception as e:
                    # recover from any per‐connection error
                    print(f"[OrderBook] Error: {e!r}. Reconnect in {backoff}s")
                    # if REST wait_for timed out, _heartbeat_task will close ws or fire here
                    await asyncio.sleep(backoff)
                    backoff = min(backoff*2, 30)

                finally:
                    # always cancel background tasks and close WS
                    if snapshot_future:
                        snapshot_future.cancel()
                    if heartbeat_task:
                        heartbeat_task.cancel()
                    if self.ws:
                        await self.ws.close()
                    print("[OrderBook] Disconnected; will reconnect.")

            print("[OrderBook] stream_and_collect stopped.")

        except Exception as e:
            # fatal error in the whole collector loop
            print(f"[OrderBook] FATAL error in stream_and_collect: {e!r}")

        finally:
            # final cleanup on exit or fatal
            self.is_running = False
            if self._heartbeat_task:
                self._heartbeat_task.cancel()

    def start(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Save the loop so stop() can close the websocket
        self.loop = loop

        try:
            loop.run_until_complete(self.stream_and_collect())
        finally:
            loop.close()

    def stop(self):
        """
        Signal the stream loop to exit and close the websocket cleanly.
        """
        # 1) prevent further reconnect attempts
        self.is_running = False
        if self.ws:
            # force the async‐for to break immediately
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)

    async def _heartbeat(self):
        try:
            while self.is_running:
                await asyncio.sleep(self.HEARTBEAT_TIMEOUT/2)
                if time.time() - self._last_msg_time > self.HEARTBEAT_TIMEOUT:
                    print("[OrderBook] Heartbeat timeout; forcing reconnect")
                    await self.ws.close()
                    return
        except asyncio.CancelledError:
            return





#########################################
# Trades Collector with Aggregated Snapshot Bars (Synchronized)
#########################################

class CopilotTradesCollector:
    def __init__(self, save_to_file=False, snapshot_interval=1.0):
        # optional queue to emit each tick's price instantly
        self.output_price_queue = None
        self.ws_url = "wss://fstream.binance.com/ws/btcusdt@trade"
        self.ws = None
        self.is_running = False
        self.save_to_file = save_to_file
        self.snapshot_interval = snapshot_interval
        self.trades_buffer = []
        self.trades_bars = []
        self.trade_count = 0
        self.total_volume = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.start_time = None
        self.snapshot_thread = None
        self.sync_base = None  # set externally by GUI
        # Lock to protect shared buffers across threads
        self._lock = threading.Lock()
        # Event to signal snapshot_loop to stop immediately
        self._stop_event = threading.Event()

    def on_open(self, ws):
        print(f"[Trades {datetime.now().strftime('%H:%M:%S')}] Connected to Binance Trade Stream")
        self.is_running = True
        self.start_time = time.time()
        if self.sync_base is None:
            self.sync_base = time.time()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            trade_time = datetime.fromtimestamp(data['T'] / 1000)
            trade_info = {
                'timestamp': datetime.now().isoformat(),
                'trade_time': trade_time.isoformat(),
                'price': float(data['p']),
                'quantity': float(data['q']),
                'side': 'SELL' if data['m'] else 'BUY',
                'trade_id': data.get('t'),
                'is_buyer_maker': data['m'],
                'value_usd': float(data['p']) * float(data['q'])
            }
            # 1) IMMEDIATELY emit the last price
            if self.output_price_queue is not None:
                try:
                    # push raw price (float) to GUI
                    self.output_price_queue.put(trade_info['price'], block=False)
                except queue.Full:
                    pass
            # thread‐safe update of buffer & counters
            with self._lock:
                self.trades_buffer.append(trade_info)
                self.trade_count += 1
                self.total_volume += trade_info['quantity']
                if trade_info['side'] == 'BUY':
                    self.buy_volume += trade_info['quantity']
                else:
                    self.sell_volume += trade_info['quantity']
        except Exception as e:
            print(f"[Trades] Error processing message: {e}")

    def on_error(self, ws, error):
        print(f"[Trades] WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[Trades {datetime.now().strftime('%H:%M:%S')}] WebSocket connection closed")


    def snapshot_loop(self):
        """
        This loop aggregates trades into a bar every snapshot_interval seconds.
        """
        try:
            while not self._stop_event.is_set():
                # align to the next boundary
                current = time.time()
                n = int((current - self.sync_base) / self.snapshot_interval) + 1
                boundary = self.sync_base + n * self.snapshot_interval
                sleep_time = boundary - time.time()
                if sleep_time > 0:
                    # wait but wake immediately if stop_event set
                    if self._stop_event.wait(sleep_time):
                        break

                # snapshot_time is the ISO timestamp we plan to assign
                snapshot_time = datetime.fromtimestamp(boundary).isoformat()

                # Atomically grab & clear the buffer
                with self._lock:
                    buffer_copy = list(self.trades_buffer)
                    self.trades_buffer.clear()

                # build the bar
                if buffer_copy:
                    total_vol     = sum(t["quantity"] for t in buffer_copy)
                    buy_vol       = sum(t["quantity"] for t in buffer_copy if t["side"] == "BUY")
                    sell_vol      = sum(t["quantity"] for t in buffer_copy if t["side"] == "SELL")
                    closing_price = buffer_copy[-1]["price"]
                else:
                    # no trades: carry forward last close if available
                    with self._lock:
                        closing_price = (
                            self.trades_bars[-1]["close_price"]
                            if self.trades_bars else 0.0
                        )
                    total_vol = buy_vol = sell_vol = 0.0

                bar = {
                    "bar_time": snapshot_time,
                    "close_price": closing_price,
                    "total_volume": total_vol,
                    "buy_volume": buy_vol,
                    "sell_volume": sell_vol
                }

                # (debug logging removed)

                # Append new bar under lock and push to queuepy
                with self._lock:
                    self.trades_bars.append(bar)
                if hasattr(self, "output_queue"):
                    self.output_queue.put(bar)

        except Exception as e:
            print(f"[TradesCollector] snapshot_loop error: {e}")



    def start(self):
        """
        Launch snapshot thread and keep WS running with reconnect/backoff.
        """
        self.is_running = True
        # 1) start bar‐aggregation loop
        self.snapshot_thread = threading.Thread(target=self.snapshot_loop, daemon=True)
        self.snapshot_thread.start()

        # 2) connect + reconnect with backoff, rely on run_forever's ping/pong
        backoff = 1
        while self.is_running:
            # create WS without manual ping handlers
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            print("[Trades] Connecting to WS…")
            # automatically send PING every 30s, drop if no PONG in 10s
            self.ws.run_forever(ping_interval=30, ping_timeout=10)

            # always reconnect unless stop() flips is_running to False
            print(f"[Trades] Disconnected. Reconnecting in {backoff}s…")
            time.sleep(backoff)
            backoff = min(backoff*2, 30)
        print("[Trades] start() loop exiting.")

    def stop(self):
        """
        Break out of the reconnect loop and close the WS.
        """
        # 1) signal snapshot_loop to exit immediately
        self._stop_event.set()
        # 2) stop the reconnect loop
        self.is_running = False

        # 2) if websocket is open, close it to unblock run_forever()
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass


#########################################
# Combined GUI with User-Defined Snapshot Interval and Rolling Window for Signals (Scaling Only Selected Metrics)
#########################################

class CombinedGUI(tk.Tk):
    def __init__(self, default_snapshot_sec=10.0):
        super().__init__()
        self.title("Copilot Combined Data GUI")
        self.geometry("740x1080") # window size

        # separate lookbacks for TIR parameter smoothing
        self.tir_sma_window = 5
        self.tir_ema_window = 5

        # queue for instant‐price updates
        self.price_queue   = queue.Queue()
        self.refresh_interval_ms = int(default_snapshot_sec * 1000)
        self.orderbook_collector = OrderBookCollector(snapshot_interval=default_snapshot_sec)
        self.trades_collector = CopilotTradesCollector(save_to_file=False, snapshot_interval=default_snapshot_sec)
        # wire the collector to our new queue
        self.trades_collector.output_price_queue = self.price_queue
        self.orderbook_thread = None
        self.trades_thread = None
        self.data_running = False
        # User-defined rolling window (# snapshots)
        self.rolling_window = 6

        # --- Initialize rolling histories & EMA states so compute_metrics never sees undefined attrs ---
        from collections import deque
        # deque of raw TIR values for SMA
        self.tir_history            = deque(maxlen=self.tir_sma_window)
        self.combined_tir_history   = deque(maxlen=self.rolling_window)

        self.ctvd_history           = deque(maxlen=self.rolling_window)
        self.ctvd_norm_history      = deque(maxlen=30)

        self.wbair_history          = deque(maxlen=self.rolling_window)
        # shorten z-score window for quicker re-normalization
        self.wbair_dynamic_history  = deque(maxlen=30) 

        self.slfi_history           = deque(maxlen=self.rolling_window)
        self.slfi_norm_history      = deque(maxlen=50)

        # history of raw add‐imbalance for zero‐centering
        self.raw_add_history        = deque(maxlen=self.rolling_window)
        # history of raw cancel‐imbalance for zero‐centering
        self.raw_can_history        = deque(maxlen=self.rolling_window)

        # history of z-scored add‐imbalance for smoothing
        self.add_z_history          = deque(maxlen=self.rolling_window)
        # history of z-scored cancel‐imbalance for smoothing
        self.can_z_history          = deque(maxlen=self.rolling_window)        

        # PriceMOM: cumulative bar-to-bar price change momentum
        self.pricemom_history        = deque(maxlen=self.rolling_window)
        self.pricemom_norm_history   = deque(maxlen=Config.CTVD_NORM_HISTORY_MAX)
        self.raw_pricemom_cumulative = 0.0
        self.ema_pricemom            = None

        # EMA state placeholders
        self.ema_tir         = None
        self.ema_ctvd        = None
        self.ema_wbair       = None
        self.ema_slfi_bid    = None
        self.ema_slfi_ask    = None
        self.ema_slfi_filtered = None
        # previous Open Interest to compute per-interval delta
        self.prev_oi = None
        # raw orderbook from previous snapshot (for re-binning)
        self.prev_raw_bids = None
        self.prev_raw_asks = None
        # state for Cancellation vs. Addition ratio
        self.prev_canc_add_ratio   = None
        # EMA state for side‐specific add/cancel imbalances
        self.prev_add_imbalance    = None
        self.prev_cancel_imbalance = None


        # Start the TCP streamer in a background thread.
        self.tcp_streamer = MetricsTCPStreamer(host="127.0.0.1", port=9999)
        self.tcp_streamer.start()
        # history_full will flip True after ~30 min

        # ---- set up shutdown event and thread-safe queues for signal pipeline ----
        self.stop_event = threading.Event()
        self.orderbook_queue = queue.Queue()
        self.trades_queue    = queue.Queue()
        self.metrics_queue   = queue.Queue()     # for the GUI
        self.broadcast_queue = queue.Queue()     # for the TCP broadcaster
        self.latest_metrics  = None

        # hand each collector its input queue
        self.orderbook_collector.output_queue = self.orderbook_queue
        self.trades_collector.output_queue    = self.trades_queue

        # start the background signal‐worker (save handle for join)
        self.signal_thread = threading.Thread(target=self.signal_worker, daemon=True)
        self.signal_thread.start()

        # start the dedicated broadcaster thread
        self.broadcaster_thread = threading.Thread(target=self._broadcaster, daemon=True)
        self.broadcaster_thread.start()

        self.create_widgets()
        # schedule the first GUI update at fixed interval
        self.after(self.refresh_interval_ms, self._run_gui_update)


    def _broadcaster(self):
        """
        Dedicated thread: read from metrics_queue and broadcast over TCP.
        """
        while not self.stop_event.is_set():
            try:
                # non‐blocking, wake every 10 ms if no metrics
                metrics = self.broadcast_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue

            try:
                self.tcp_streamer.broadcast(metrics)
            except Exception as e:
                print(f"[Broadcaster] Error broadcasting metrics: {e}")


    def close_app(self):
        # 1) Stop background pipeline
        self.stop_event.set()

        # 2) Stop collectors
        try:
            self.orderbook_collector.stop()
        except:
            pass
        try:
            self.trades_collector.stop()
        except:
            pass

        # 3) Stop TCP streamer
        if hasattr(self, "tcp_streamer"):
            self.tcp_streamer.stop()

        # 4) Join threads (short timeouts)
        if hasattr(self, "signal_thread"):
            self.signal_thread.join(timeout=1.0)
        if self.orderbook_thread and self.orderbook_thread.is_alive():
            self.orderbook_thread.join(timeout=1.0)
        if self.trades_thread and self.trades_thread.is_alive():
            self.trades_thread.join(timeout=1.0)
        if hasattr(self, "tcp_streamer") and self.tcp_streamer.is_alive():
            self.tcp_streamer.join(timeout=1.0)

        # 5) Finally close the GUI
        self.destroy()


    def create_widgets(self):
        # Row 1: Start/Stop | Status | History Buffer
        row1 = tk.Frame(self)
        row1.pack(side="top", fill="x", padx=10, pady=(5,0))

        self.start_stop_btn = tk.Button(
            row1, text="Start Collection",
            command=self.toggle_collection,
            width=15, font=("Arial", 10)
        )
        self.start_stop_btn.pack(side="left", padx=5)

        self.status_label = tk.Label(
            row1, text="Status: Stopped",
            font=("Arial", 10, "bold"), fg="#AA0000"
        )
        self.status_label.pack(side="left", padx=5)


        # Row 2: Snapshot interval | Rolling window
        row2 = tk.Frame(self)
        row2.pack(side="top", fill="x", padx=10, pady=(5,0))

        self.snapshot_var = tk.StringVar(value="10.0") # snapshot default
        tk.Label(
            row2, text="Snapshot Interval (sec):",
            font=("Arial", 10)
        ).pack(side="left", padx=(0,5))

        self.snapshot_entry = tk.Entry(
            row2, textvariable=self.snapshot_var,
            width=5, font=("Arial", 10)
        )
        self.snapshot_entry.pack(side="left", padx=(0,15))

        self.rolling_window_var = tk.StringVar(value="6") # rolling window default
        tk.Label(
            row2, text="Rolling Window (# snapshots):",
            font=("Arial", 10)
        ).pack(side="left", padx=(0,5))

        self.rolling_window_entry = tk.Entry(
            row2, textvariable=self.rolling_window_var,
            width=5, font=("Arial", 10)
        )
        self.rolling_window_entry.pack(side="left")

        # Row 3: Last Price
        row3 = tk.Frame(self)
        row3.pack(side="top", fill="x", padx=10, pady=(5,10))

        self.price_label = tk.Label(
            row3, text="Last Price: --",
            font=("Arial", 10)
        )
        self.price_label.pack(side="left")
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side="top", fill="both", expand=True)
        self.orderbook_frame = tk.Frame(self.notebook)
        self.notebook.add(self.orderbook_frame, text="Order Book")
        self.create_orderbook_widgets(self.orderbook_frame)
        self.trades_frame = tk.Frame(self.notebook)
        self.notebook.add(self.trades_frame, text="Trades")
        self.create_trades_widgets(self.trades_frame)
        self.signals_frame = tk.Frame(self.notebook)
        self.notebook.add(self.signals_frame, text="Signals")
        self.create_signals_widgets(self.signals_frame)

        # Logs tab (capture all print()s here)
        self.logs_frame = tk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="Logs")
        self.create_logs_widgets(self.logs_frame)

    def create_signals_widgets(self, parent):
        self.signals_text = tk.Text(parent, font=("Courier", 11), wrap="word",
                                     state="disabled", bg="#1E1E1E", fg="#D4D4D4")
        self.signals_text.pack(fill="both", expand=True, padx=10, pady=10)


    # >>>>>>>>   METRICS CALCULATIONS   <<<<<<<<<<<
    # 
    def compute_metrics(self):
        # Latest trade bar and volume metrics
        if self.trades_collector.trades_bars:
            bar = self.trades_collector.trades_bars[-1]
        else:
            bar = {
                "close_price":  0.0,
                "total_volume": 0.0,
                "buy_volume":   0.0,
                "sell_volume":  0.0
            }
        results = {
            "Price":        float(bar.get("close_price",  0.0)),
            "total_volume": float(bar.get("total_volume", 0.0)),
            "buy_volume":   float(bar.get("buy_volume",   0.0)),
            "sell_volume":  float(bar.get("sell_volume",  0.0)),
        }

        # TIR
        tir, self.ema_tir = compute_tir(
            self.trades_collector.trades_bars,
            self.tir_sma_window,
            self.tir_ema_window,
            self.tir_history,
            self.ema_tir,
            self.combined_tir_history
        )
        results["scaled_TIR"] = tir

        # CTVD
        ctvd, self.ema_ctvd = compute_ctvd(
            self.trades_collector.trades_bars,
            self.rolling_window,
            self.ctvd_history,
            self.ema_ctvd,
            self.ctvd_norm_history
        )
        results["CTVD"] = ctvd


        # Price Momentum (PriceMOM)
        bars = self.trades_collector.trades_bars
        if len(bars) >= 2:
            delta_price = bars[-1].get("close_price", 0.0) \
                        - bars[-2].get("close_price", 0.0)
        else:
            delta_price = 0.0
        # damp each tick’s contribution
        self.raw_pricemom_cumulative += Config.PRICEMOM_DAMPING * delta_price
        raw_pm = self.raw_pricemom_cumulative
        # SMA
        self.pricemom_history.append(raw_pm)
        s_pm = sma(self.pricemom_history)
        # EMA
        alpha_pm = Config.CTVD_EMA_WEIGHT * (2.0 / (self.rolling_window + 1))
        base_pm  = raw_pm if self.ema_pricemom is None else self.ema_pricemom
        e_pm      = ema(raw_pm, base_pm, alpha_pm)
        comb_pm   = 0.4 * s_pm + 0.6 * e_pm
        self.pricemom_norm_history.append(comb_pm)
        z_pm      = zscore(comb_pm, self.pricemom_norm_history)
        # use a gentler arctan mapping for PriceMOM
        price_mom = arctan_norm(z_pm, Config.PRICEMOM_ARCTAN_K)
        self.ema_pricemom = e_pm
        results["PriceMOM"] = 1.10 * price_mom # scaling pricemom


        # WBAIR
        wbair, self.ema_wbair = compute_wbair(
            self.orderbook_collector.latest_snapshot,
            self.rolling_window,
            self.wbair_history,
            self.ema_wbair,
            self.wbair_dynamic_history
        )
        results["scaled_WBAIR"] = wbair

        # SLFI
        slfi, b, a, f = compute_slfi(
            self.orderbook_collector.latest_snapshot,
            self.rolling_window,
            self.slfi_history,
            self.ema_slfi_bid,
            self.ema_slfi_ask,
            self.ema_slfi_filtered,
            self.slfi_norm_history
        )
        self.ema_slfi_bid      = b
        self.ema_slfi_ask      = a
        self.ema_slfi_filtered = f
        results["scaled_SLFI"] = slfi

        # ─── Gaussian‐Weighted Add‐Imbalance (fixed‐grid + z‐score zero‐centering) ───
        add_bid = add_ask = can_bid = can_ask = 0.0
        snapshot = self.orderbook_collector.latest_snapshot or {}
        # use fixed bin_center to align prev & curr snapshots
        center    = snapshot.get("bin_center", snapshot.get("last_price", 0))
        prev_bins = self.orderbook_collector.aggregate_bins(
            last_price=center,
            bids=self.prev_raw_bids  or {},
            asks=self.prev_raw_asks  or {}
        )
        curr_bins = self.orderbook_collector.aggregate_bins(
            last_price=center,
            bids=snapshot.get("raw_bids", {}),
            asks=snapshot.get("raw_asks", {})
        )
        near   = Config.ADD_IMB_NEAR_BINS
        sigma  = near * Config.ADD_IMB_SIGMA_FACTOR
        mid_idx     = len(prev_bins)//2
        for i, (pb, cb) in enumerate(zip(prev_bins, curr_bins)):
            if abs(i - mid_idx) > near:
                continue
            w = math.exp(-((i - mid_idx)**2) / (2*sigma**2))
            for side in ("bid_qty","ask_qty"):
                diff = cb.get(side, 0.0) - pb.get(side, 0.0)
                if diff > 0:
                    if side=="bid_qty": add_bid += diff*w
                    else:               add_ask += diff*w
                else:
                    if side=="bid_qty": can_bid += -diff*w
                    else:               can_ask += -diff*w

        # choose raw imbalance: absolute vs relative diff
        eps = 1e-6
        if Config.ADD_IMB_USE_RELATIVE:
            rel_add = rel_can = 0.0
            for i, (pb, cb) in enumerate(zip(prev_bins, curr_bins)):
                if abs(i - mid_idx) > near:
                    continue
                w = math.exp(-((i - mid_idx)**2) / (2 * sigma**2))
                # net depth per bin
                prev_net = pb.get("bid_qty",0) - pb.get("ask_qty",0)
                curr_net = cb.get("bid_qty",0) - cb.get("ask_qty",0)
                # normalize by bin liquidity
                avg_vol = (
                    (pb.get("bid_qty",0) + pb.get("ask_qty",0)) +
                    (cb.get("bid_qty",0) + cb.get("ask_qty",0))
                ) / 2.0
                if avg_vol > Config.ADD_IMB_REL_THRESHOLD:
                    delta = curr_net - prev_net
                    if delta > 0:
                        rel_add += w * (delta / avg_vol)
                    else:
                        rel_can += w * (-delta / avg_vol)
            raw_iadd = (rel_add - rel_can) / (rel_add + rel_can + eps)
        else:
            raw_iadd = (add_bid - add_ask) / (add_bid + add_ask + eps)

        # now z‐score + scale + EMA as before
        self.raw_add_history.append(raw_iadd)
        mean = sum(self.raw_add_history)/len(self.raw_add_history)
        var  = sum((x-mean)**2 for x in self.raw_add_history)/len(self.raw_add_history)
        std  = math.sqrt(var) if var>0 else 1e-6
        raw_iadd_z = (raw_iadd - mean) / std

        self.add_z_history.append(raw_iadd_z)
        # SG-filter on the z-scored history
        if len(self.add_z_history) >= Config.ADD_IMB_FILTER_WINDOW:
            filt_w = Config.ADD_IMB_FILTER_WINDOW
            if filt_w % 2 == 0:
                filt_w += 1
            filt_w = min(filt_w, len(self.add_z_history))
            raw_iadd_z = savgol_filter(np.array(self.add_z_history), filt_w, 1)[-1]

        # scale + EMA for Add Imbalance
        scaled_iadd = 100*math.atan(raw_iadd_z/2.0)/(math.pi/2)
        alpha       = 2.0/(self.rolling_window+1)
        base        = self.prev_add_imbalance if self.prev_add_imbalance is not None else scaled_iadd
        e_iadd      = alpha*scaled_iadd + (1-alpha)*base
        self.prev_add_imbalance = e_iadd
        results["AddImb"]      = e_iadd

        # ─── Cancel Imbalance (z-score + EMA) ───
        raw_ican = (can_ask - can_bid) / (can_ask + can_bid + eps)
        self.raw_can_history.append(raw_ican)
        mean_c = sum(self.raw_can_history) / len(self.raw_can_history)
        var_c  = sum((x-mean_c)**2 for x in self.raw_can_history) / len(self.raw_can_history)
        std_c  = math.sqrt(var_c) if var_c > 0 else 1e-6
        raw_ican_z  = (raw_ican - mean_c) / std_c

        self.can_z_history.append(raw_ican_z)
        # SG-filter on the z-scored history (symmetric with AddImb)
        if len(self.can_z_history) >= Config.ADD_IMB_FILTER_WINDOW:
            filt_w = Config.ADD_IMB_FILTER_WINDOW
            if filt_w % 2 == 0:
                filt_w += 1
            filt_w = min(filt_w, len(self.can_z_history))
            raw_ican_z = savgol_filter(np.array(self.can_z_history), filt_w, 1)[-1]

        scaled_ican = 100*math.atan(raw_ican_z/2.0)/(math.pi/2)
        base_can    = self.prev_cancel_imbalance if self.prev_cancel_imbalance is not None else scaled_ican
        e_ican      = alpha*scaled_ican + (1-alpha)*base_can
        self.prev_cancel_imbalance = e_ican
        results["CancelImb"] = e_ican

        # ——— Cancellation vs. Addition Rate ——————
        cancel = add = 0.0
        if snapshot.get("bins") and self.prev_raw_bids is not None:
            aligned_prev = self.orderbook_collector.aggregate_bins(
                last_price=snapshot["last_price"],
                bids=self.prev_raw_bids,
                asks=self.prev_raw_asks
            )
            for pb, cb in zip(aligned_prev, snapshot["bins"]):
                for side in ("bid_qty", "ask_qty"):
                    diff = cb.get(side, 0.0) - pb.get(side, 0.0)
                    if diff > 0:
                        add += diff
                    else:
                        cancel += -diff
        raw_ratio = cancel/add if add > 0 else 0.0
        alpha   = 2.0/(self.rolling_window+1)
        base    = raw_ratio if self.prev_canc_add_ratio is None else self.prev_canc_add_ratio
        e_ratio = alpha*raw_ratio + (1-alpha)*base
        self.prev_canc_add_ratio = e_ratio
        results["CA_Ratio"] = e_ratio

        # ─── Finally, store this snapshot’s raw book for use in the next cycle ───
        self.prev_raw_bids = snapshot.get("raw_bids")
        self.prev_raw_asks = snapshot.get("raw_asks")


        # ——— Delta Open Interest per snapshot interval ———
        try:
            oi_val = float(requests.get(
                "https://fapi.binance.com/fapi/v1/openInterest",
                params={"symbol": "BTCUSDT"}, timeout=2.0
            ).json().get("openInterest", 0.0))
        except Exception:
            oi_val = self.prev_oi if self.prev_oi is not None else 0.0
        # compute delta vs last interval
        if self.prev_oi is None:
            delta_oi = 0.0
        else:
            delta_oi = oi_val - self.prev_oi
        self.prev_oi = oi_val
        results["deltaOI"] = delta_oi

        return results
    # 
    # >>>>>>>>   END METRICS CALCULATIONS   <<<<<<<<<<


    def create_orderbook_widgets(self, parent):
        frame_main = tk.Frame(parent, borderwidth=0, highlightthickness=0)
        frame_main.pack(fill="both", expand=True, padx=10, pady=0, ipady=0)
        self.asks_frame = tk.Frame(frame_main, borderwidth=0, highlightthickness=0)
        self.asks_frame.pack(side="top", fill="x", pady=0, ipady=0)
        self.mid_price_label = tk.Label(frame_main, text="Mid Price: N/A", font=("Arial", 10, "bold"), anchor="w", borderwidth=0, highlightthickness=0, padx=0, pady=0)
        self.mid_price_label.pack(side="top", fill="x", pady=1, ipady=0)
        self.bids_frame = tk.Frame(frame_main, borderwidth=0, highlightthickness=0)
        self.bids_frame.pack(side="top", fill="x", pady=0, ipady=0)
        self.ask_labels = []
        for i in range(32):
            lbl = tk.Label(self.asks_frame, text="", anchor="w", font=("Courier", 8), borderwidth=0, highlightthickness=0, padx=0, pady=0)
            lbl.pack(fill="x", pady=0, ipady=0)
            self.ask_labels.append(lbl)
        self.bid_labels = []
        for i in range(32):
            lbl = tk.Label(self.bids_frame, text="", anchor="w", font=("Courier", 8), borderwidth=0, highlightthickness=0, padx=0, pady=0)
            lbl.pack(fill="x", pady=0, ipady=0)
            self.bid_labels.append(lbl)

    def create_trades_widgets(self, parent):
        self.trades_stats_label = tk.Label(parent, text="Trades Stats: Loading...", font=("Arial", 10))
        self.trades_stats_label.pack(pady=5)
        columns = ("bar_time", "close_price", "total_volume", "buy_volume", "sell_volume")

        # container for tree + scrollbar
        trades_container = tk.Frame(parent, borderwidth=0, highlightthickness=0)
        trades_container.pack(fill="both", expand=True, padx=10, pady=5)

        # compact treeview style
        style = ttk.Style(trades_container)
        style.configure("Trades.Treeview", rowheight=16, font=("Courier", 9))
        style.configure("Trades.Treeview.Heading", font=("Arial", 10))

        self.trades_tree = ttk.Treeview(trades_container, columns=columns, show="headings", height=10, style="Trades.Treeview")
        self.trades_tree.heading("bar_time", text="Bar Time")
        self.trades_tree.heading("close_price", text="Close Price")
        self.trades_tree.heading("total_volume", text="Total Volume")
        self.trades_tree.heading("buy_volume", text="Buy Volume")
        self.trades_tree.heading("sell_volume", text="Sell Volume")
        for col in columns:
            self.trades_tree.column(col, width=120, anchor="center")

        # vertical scrollbar
        trades_vsb = ttk.Scrollbar(trades_container, orient="vertical", command=self.trades_tree.yview)
        self.trades_tree.configure(yscrollcommand=trades_vsb.set)

        self.trades_tree.pack(side="left", fill="both", expand=True)
        trades_vsb.pack(side="right", fill="y")



    def update_signals_tab(self):
        metrics = self.compute_metrics()
        text = (
            f"Price:            {metrics.get('Price',        0):.2f}\n\n"
            f"PriceMOM:         {metrics.get('PriceMOM',     0):.2f}\n\n"
            f"Total Volume:     {metrics.get('total_volume', 0):.4f} BTC\n"
            f"Buy Volume:       {metrics.get('buy_volume',   0):.4f} BTC\n"
            f"Sell Volume:      {metrics.get('sell_volume',  0):.4f} BTC\n\n"
            f"WBAIR:            {metrics.get('scaled_WBAIR', 0):.2f}\n\n"
            f"SLFI:             {metrics.get('scaled_SLFI',  0):.2f}\n\n"
            f"TIR:              {metrics.get('scaled_TIR',   0):.2f}\n\n"
            f"CTVD:             {metrics.get('CTVD',        0):.2f}\n\n"
            f"Delta OI ({self.snapshot_var.get()}s): {metrics.get('deltaOI', 0):.2f}\n\n"
            f"Add Imbalance:    {metrics.get('AddImb',    0): .3f}\n\n"
            f"Cancel Imbalance: {metrics.get('CancelImb', 0): .3f}\n\n"
            f"CA Ratio:         {metrics.get('CA_Ratio',    0):.3f}"
        )
        self.signals_text.config(state="normal")
        self.signals_text.delete("1.0", "end")
        self.signals_text.insert("end", text)
        self.signals_text.config(state="disabled")



    def update_gui(self):
        # 0) Instant‐price update (zero‐lag)
        try:
            while True:
                px = self.price_queue.get_nowait()
                self.price_label.config(text=f"Last Price: ${px:.2f}")
        except queue.Empty:
            pass

        # pull any newly computed metrics
        try:
            while True:
                self.latest_metrics = self.metrics_queue.get_nowait()
        except queue.Empty:
            pass

        status_text = "Status: "

        if self.orderbook_collector.is_running and self.trades_collector.is_running:
            status_text += "Connected"
        else:
            status_text += "Disconnected"
        # update status text and color
        if "Connected" in status_text:
            self.status_label.config(text=status_text, fg="#00AA00")
        else:
            self.status_label.config(text=status_text, fg="#AA0000")

        snapshot = self.orderbook_collector.latest_snapshot
        if snapshot and snapshot.get("last_price") and snapshot.get("bins") and len(snapshot["bins"]) >= 64:
            last_price = snapshot["last_price"]
            self.mid_price_label.config(text=f"Mid Price: ${last_price:.2f}")
            ask_bins = snapshot["bins"][32:64][::-1]
            bid_bins = snapshot["bins"][0:32][::-1]
            for i, bin_data in enumerate(ask_bins):
                r_start, r_end = bin_data.get("bin_range", [0, 0])
                ask_qty = bin_data.get("ask_qty", 0.0)
                self.ask_labels[i].config(
                    text=f"{r_start:,.2f}-{r_end:,.2f} | Ask Qty: {ask_qty:.4f}"
                )
            for i, bin_data in enumerate(bid_bins):
                r_start, r_end = bin_data.get("bin_range", [0, 0])
                bid_qty = bin_data.get("bid_qty", 0.0)
                self.bid_labels[i].config(
                    text=f"{r_start:,.2f}-{r_end:,.2f} | Bid Qty: {bid_qty:.4f}"
                )

        total_trades = self.trades_collector.trade_count
        total_vol = self.trades_collector.total_volume
        buy_vol = self.trades_collector.buy_volume
        sell_vol = self.trades_collector.sell_volume
        if self.data_running and self.trades_collector.start_time:
            runtime = time.time() - self.trades_collector.start_time
            trades_per_sec = total_trades / runtime if runtime > 0 else 0
        else:
            trades_per_sec = 0
        stats_text = (
            f"Trades: {total_trades} | Trades/sec: {trades_per_sec:.2f} | "
            f"Total Vol: {total_vol:.4f} BTC | Buy Vol: {buy_vol:.4f} BTC | "
            f"Sell Vol: {sell_vol:.4f} BTC"
        )
        self.trades_stats_label.config(text=stats_text)
        self.trades_tree.delete(*self.trades_tree.get_children())
        bars = self.trades_collector.trades_bars
        if len(bars) > 10:
            bars = bars[-10:]
        for bar in reversed(bars):
            self.trades_tree.insert(
                "", "end",
                values=(
                    bar.get("bar_time", ""),
                    f"{bar.get('close_price', 0):.2f}",
                    f"{bar.get('total_volume', 0):.4f}",
                    f"{bar.get('buy_volume', 0):.4f}",
                    f"{bar.get('sell_volume', 0):.4f}"
                )
            )

        # update the Signals tab
        self.update_signals_tab()



    def create_logs_widgets(self, parent):
        """ScrollText that shows every print() from this process."""
        self.log_widget = ScrolledText(
            parent,
            state="disabled",
            bg="black",
            fg="white",
            font=("Courier", 10)
        )
        self.log_widget.pack(fill="both", expand=True, padx=10, pady=10)
        # tag colors
        self.log_widget.tag_config("stdout", foreground="white")
        self.log_widget.tag_config("stderr", foreground="red")
        # redirect sys.stdout & sys.stderr into this widget
        sys.stdout = TextRedirector(self.log_widget, "stdout")
        sys.stderr = TextRedirector(self.log_widget, "stderr")


    def signal_worker(self):
        """
        Background thread: BLOCKING read of one OB snapshot + one trade‐bar per cycle,
        compute_metrics() immediately, then enqueue & broadcast.
        Decouple: block only on trades, not on orderbook.
        """
        local_bars = []
        while not self.stop_event.is_set():
            try:
                # 1) BLOCK until the next trade bar arrives (with timeout)
                bar = self.trades_queue.get(timeout=1.0)

                # 2) Optionally pull the latest OB if queued
                try:
                    ob = self.orderbook_queue.get_nowait()
                    self.orderbook_collector.latest_snapshot = ob
                except queue.Empty:
                    pass

                # 3) Update trade bars kept for GUI and metrics
                # Keep at least 10 bars for the Trades tab, even if rolling_window is smaller
                local_bars.append(bar)
                max_keep = max(self.rolling_window, 10)
                if len(local_bars) > max_keep:
                    local_bars = local_bars[-max_keep:]
                self.trades_collector.trades_bars = list(local_bars)

                # 4) Compute & broadcast metrics
                metrics = self.compute_metrics()
                metrics['ts'] = time.time()
                self.metrics_queue.put(metrics)
                self.broadcast_queue.put(metrics)

            except queue.Empty:
                # no bar arrived in 1s → loop again
                continue
            except Exception as e:
                print(f"[SignalWorker] compute_metrics error: {e}")

    def toggle_collection(self):
        if not self.data_running:
            self.start_collection()
            self.start_stop_btn.config(text="Stop Collection")
            self.snapshot_entry.config(state="disabled")
            self.rolling_window_entry.config(state="disabled")
        else:
            self.stop_collection()
            self.start_stop_btn.config(text="Start Collection")
            self.snapshot_entry.config(state="normal")
            self.rolling_window_entry.config(state="normal")

    def _run_gui_update(self):
        """
        Single-shot GUI update at fixed intervals.
        """
        try:
            self.update_gui()
        except Exception as e:
            print(f"[GUI] update_gui error: {e}")
        # schedule next tick
        self.after(self.refresh_interval_ms, self._run_gui_update)


# Remove any .after_idle calls in signal_worker—GUI is on fixed schedule now.


    def start_collection(self):
        try:
            snap_val = float(self.snapshot_var.get())
        except ValueError:
            snap_val = 1.0
        try:
            roll_val = int(self.rolling_window_var.get())
        except ValueError:
            roll_val = 10
        self.orderbook_collector.snapshot_interval = snap_val
        self.trades_collector.snapshot_interval = snap_val
        self.refresh_interval_ms = int(snap_val * 1000)
        self.rolling_window = roll_val
        # Clear previous rolling histories and reset EMA state
        self.tir_history.clear()
        self.ctvd_history.clear()
        self.slfi_history.clear()
        self.wbair_history.clear()
        self.ema_tir = None
        self.ema_wbair = None
        self.ema_slfi_bid = None
        self.ema_slfi_ask = None
        self.prev_bid_total = None
        self.prev_ask_total = None
        # re‐init raw‐imbalance histories to match new window
        self.raw_add_history = deque(maxlen=self.rolling_window)
        self.raw_can_history = deque(maxlen=self.rolling_window)

        # ALSO clear any normalization/“dynamic” histories so zscore() can work again
        self.ctvd_norm_history.clear()
        self.slfi_norm_history.clear()
        self.wbair_dynamic_history.clear()
        sync_base = time.time()
        self.orderbook_collector.sync_base = sync_base
        self.trades_collector.sync_base = sync_base

        self.add_z_history = deque(maxlen=50)
        self.can_z_history = deque(maxlen=50)

        if not self.orderbook_thread or not self.orderbook_thread.is_alive():
            self.orderbook_thread = threading.Thread(target=self.orderbook_collector.start, daemon=True)
            self.orderbook_thread.start()
        if not self.trades_thread or not self.trades_thread.is_alive():
            self.trades_thread = threading.Thread(target=self.trades_collector.start, daemon=True)
            self.trades_thread.start()
        self.data_running = True

    def stop_collection(self):
        self.orderbook_collector.stop()
        self.trades_collector.stop()
        self.data_running = False

# Create a shared GUI instance for import OR direct run
gui = CombinedGUI(default_snapshot_sec=10.0) # snapshot window default
gui.protocol("WM_DELETE_WINDOW", gui.close_app)

if __name__ == "__main__":
    gui.mainloop()

