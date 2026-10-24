"""Readable desktop interface for OpenBMI recorded-trial replay."""
import ctypes
import queue
import socket
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from receiver import BCIProcessor, DEFAULT_MODEL, LABELS, load_trials, receive_trial

# 启用高分屏 DPI 感知，彻底解决高分屏下文字发糊问题
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class OpenBMIApp:
    def __init__(self, root):
        self.root = root
        root.title("OpenBMI 脑电识别回放系统")
        root.geometry("1280x900")
        root.minsize(1000, 750)
        root.configure(bg="#f3f7f8")
        
        self.font_size = 11
        self.ui_scale = 1.0
        self._resize_job = None
        self._last_window_size = None
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.connection = None
        self.fullscreen = False
        self.correct = self.count = 0
        
        self.model_path = tk.StringVar(value=str(DEFAULT_MODEL))
        self.data_path = tk.StringVar()
        self.mode = tk.StringVar(value="本地文件回放")
        self.phase = tk.StringVar(value="test")
        self.speed = tk.StringVar(value="4")
        self.host = tk.StringVar(value="127.0.0.1")
        self.port = tk.StringVar(value="65432")
        self.status = tk.StringVar(value="请选择 OpenBMI MAT 文件，然后点击“开始回放”。")
        self.result = tk.StringVar(value="等待开始")
        self.detail = tk.StringVar(value="左手 / 右手二分类 · 无需重新训练")
        self.model_info = tk.StringVar(value="模型信息将在加载后显示")
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.init_styles()
        self.build()
        self._spacing = []
        self.capture_spacing(self.root)
        root.bind("<Configure>", self.schedule_resize, add="+")
        self.root.after_idle(self.resize_layout)
        
        root.bind("<F11>", self.toggle_fullscreen)
        root.bind("<Escape>", self.exit_fullscreen)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def init_styles(self):
        """极简风格与宋体全局样式配置"""
        # 使用像素字号，避免窗口缩放与 Windows DPI 缩放重复叠加。
        size = -max(11, round(self.font_size * 4 / 3))
        font_family = "宋体"
        
        self.root.option_add("*TCombobox*Listbox.font", (font_family, size))
        self.style.configure(".", font=(font_family, size))
        self.style.configure("TFrame", background="#f3f7f8")
        self.style.configure("TLabel", background="#f3f7f8", foreground="#24464d")
        
        # 极简按钮与下拉框统一样式优化，避免丑陋的系统高亮灰
        self.style.configure("TButton", padding=(round(12*self.ui_scale), round(6*self.ui_scale)), font=(font_family, size), relief="flat", background="#e8f0f2")
        self.style.configure("TEntry", font=(font_family, size))
        self.style.configure("TCombobox", font=(font_family, size), arrowsize=max(10, round(14*self.ui_scale)))
        self.style.map("TButton", background=[('active', '#d0dfe2'), ('pressed', '#c2d5d8')])
        
        # 主标题使用与看板一致的核心主题色 (#25857c)
        self.style.configure("Title.TLabel", font=(font_family, -round(23*self.ui_scale), "bold"), foreground="#25857c")
        # 正在准备 / 核心状态识别统一颜色
        self.style.configure("Result.TLabel", font=(font_family, -round(20*self.ui_scale), "bold"), foreground="#25857c")
        self.style.configure("Hint.TLabel", font=(font_family, -max(10, round(13*self.ui_scale))), foreground="#688188")
        
        self.style.configure("Treeview", rowheight=round(28*self.ui_scale), font=(font_family, size), background="#ffffff", fieldbackground="#ffffff")
        self.style.configure("Treeview.Heading", font=(font_family, size, "bold"), background="#e8f0f2", foreground="#24464d")

    def build(self):
        # 主容器：采用宽松的极简留白
        main_container = ttk.Frame(self.root, padding=24)
        self.main_container = main_container
        main_container.pack(fill="both", expand=True)
        main_container.columnconfigure(0, weight=1)
        
        # 空间权重分配：脑电图展示窗口占据绝对主体高度
        main_container.rowconfigure(0, weight=0)  # 顶部标题与操作
        main_container.rowconfigure(1, weight=0)  # 极简配置栏
        main_container.rowconfigure(2, weight=0)  # 实时看板与控制
        main_container.rowconfigure(3, weight=6)  # 脑电波形展示（主体）
        main_container.rowconfigure(4, weight=1)  # 历史表格
        main_container.rowconfigure(5, weight=0)  # 底部提示

        # 1. 顶部栏（主标题颜色与状态栏一致）
        top_bar = ttk.Frame(main_container)
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        ttk.Label(top_bar, text="OpenBMI 脑电识别回放系统", style="Title.TLabel").pack(side="left")
        ttk.Button(top_bar, text="全屏切换 (F11)", command=self.toggle_fullscreen).pack(side="right")

        # 2. 极简配置栏
        form = ttk.Frame(main_container)
        form.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        form.columnconfigure(1, weight=1)
        
        ttk.Label(form, text="模型路径").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Entry(form, textvariable=self.model_path).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(form, text="浏览...", command=self.choose_model).grid(row=0, column=2, padx=(8, 0), pady=6)

        ttk.Label(form, text="数据文件").grid(row=1, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Entry(form, textvariable=self.data_path).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(form, text="浏览...", command=self.choose_data).grid(row=1, column=2, padx=(8, 0), pady=6)

        # 参数控制行
        params = ttk.Frame(form)
        params.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        ttk.Label(params, text="运行方式").pack(side="left", padx=(0, 6))
        ttk.Combobox(params, textvariable=self.mode, values=["本地文件回放", "网络接收"], width=11, state="readonly").pack(side="left", padx=(0, 20))
        
        ttk.Label(params, text="阶段").pack(side="left", padx=(0, 6))
        ttk.Combobox(params, textvariable=self.phase, values=["test", "train"], width=6, state="readonly").pack(side="left", padx=(0, 20))
        
        ttk.Label(params, text="倍速").pack(side="left", padx=(0, 6))
        ttk.Combobox(params, textvariable=self.speed, values=["1", "2", "4", "8"], width=4, state="readonly").pack(side="left", padx=(0, 20))

        ttk.Label(params, text="IP").pack(side="left", padx=(0, 4))
        ttk.Entry(params, textvariable=self.host, width=12).pack(side="left", padx=(0, 10))
        ttk.Label(params, text="端口").pack(side="left", padx=(0, 4))
        ttk.Entry(params, textvariable=self.port, width=6).pack(side="left")
        # 保持原来的单行顺序，让输入框共同分配横向空间。
        parameter_widgets = [(widget, widget.pack_info())
                             for widget in params.winfo_children()]
        # 必须先移除所有 pack 控件，才能在同一父容器中使用 grid。
        for widget, _ in parameter_widgets:
            widget.pack_forget()
        for column, (widget, options) in enumerate(parameter_widgets):
            widget.grid(row=0, column=column, sticky="ew", padx=options.get("padx", 0))
            if isinstance(widget, (ttk.Entry, ttk.Combobox)):
                params.columnconfigure(column, weight=int(widget.cget("width")))

        # 3. 控制与实时看板区（识别动作与真实动作同行显示，字号风格统一）
        ctrl_board = ttk.Frame(main_container)
        ctrl_board.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        ctrl_board.columnconfigure(1, weight=1)

        btn_group = ttk.Frame(ctrl_board)
        btn_group.grid(row=0, column=0, sticky="w", padx=(0, 24))
        self.start_button = ttk.Button(btn_group, text="开始回放", command=self.start)
        self.start_button.pack(side="left", padx=(0, 10))
        ttk.Button(btn_group, text="停止", command=self.stop).pack(side="left")

        info_group = ttk.Frame(ctrl_board)
        info_group.grid(row=0, column=1, sticky="ew")
        
        # 同行展示区域
        row_line1 = ttk.Frame(info_group)
        self.info_group = info_group
        row_line1.pack(anchor="w", fill="x")
        row_line1.columnconfigure(1, weight=1)
        self.result_label = ttk.Label(row_line1, textvariable=self.result, style="Result.TLabel")
        self.result_label.grid(row=0, column=0, sticky="w", padx=(0, 20))
        self.detail_label = ttk.Label(row_line1, textvariable=self.detail, style="TLabel")
        self.detail_label.grid(row=0, column=1, sticky="ew")
        
        self.status_label = ttk.Label(info_group, textvariable=self.status, style="Hint.TLabel")
        self.status_label.pack(anchor="w", fill="x", pady=(4, 0))
        info_group.bind("<Configure>", self.wrap_info, add="+")

        # 4. 脑电波形展示窗口（高清抗锯齿渲染）
        plot_container = ttk.Frame(main_container)
        plot_container.grid(row=3, column=0, sticky="nsew", pady=(0, 16))
        plot_container.columnconfigure(0, weight=1)
        plot_container.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 4), dpi=120, facecolor="#f3f7f8")
        self.axes = self.figure.subplots(3, 1, sharex=True)
        self.figure.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.18, hspace=0.15)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.get_tk_widget().configure(width=1, height=1)
        self.canvas.get_tk_widget().bind("<Configure>", self.resize_plot, add="+")
        
        for ax, name in zip(self.axes, ["C3", "Cz", "C4"]):
            ax.set_facecolor("#ffffff")
            ax.set_ylabel(name + " (µV)", fontname="宋体", fontsize=9, color="#24464d")
            ax.tick_params(colors="#24464d", labelsize=8)
            ax.grid(alpha=0.15, linestyle="--", color="#688188")
            for spine in ax.spines.values():
                spine.set_color("#d0dfe2")
        self.axes[-1].set_xlabel("Time (s)", fontname="宋体", fontsize=9, color="#24464d")

        # 5. 历史记录表格区
        history_frame = ttk.Frame(main_container)
        history_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 8))
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
        self.table = ttk.Treeview(history_frame, columns=("trial", "truth", "prediction", "confidence", "time"),
                                  show="headings", height=3)
        for name, title in zip(self.table["columns"], ["试次", "真实动作", "识别动作", "预测置信度", "推理耗时"]):
            self.table.heading(name, text=title)
            self.table.column(name, width=140, minwidth=40, stretch=True, anchor="center")
        self.table.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.table.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.bind("<Configure>", self.resize_table, add="+")

    def capture_spacing(self, parent):
        """保存原始间距；每次从基准缩放，不累计误差。"""
        for widget in parent.winfo_children():
            manager = widget.winfo_manager()
            if manager in ("pack", "grid"):
                info = getattr(widget, manager + "_info")()
                pads = {key: tuple(int(float(v)) for v in self.root.tk.splitlist(info[key])
                                   ) if isinstance(info[key], (str, tuple, list)) else (int(info[key]),)
                        for key in ("padx", "pady") if key in info}
                self._spacing.append((widget, manager, pads))
            self.capture_spacing(widget)

    def schedule_resize(self, event):
        if event.widget is not self.root:
            return
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self.resize_layout)

    def resize_layout(self):
        self._resize_job = None
        dimensions = (self.root.winfo_width(), self.root.winfo_height())
        if min(dimensions) <= 1 or dimensions == self._last_window_size:
            return
        self._last_window_size = dimensions
        self.ui_scale = max(0.75, min(dimensions[0] / 1280, dimensions[1] / 900))
        self.font_size = 11 * self.ui_scale
        self.init_styles()
        self.main_container.configure(padding=round(24*self.ui_scale))
        for widget, manager, pads in self._spacing:
            getattr(widget, manager + "_configure")(**{
                key: tuple(round(value*self.ui_scale) for value in values)
                for key, values in pads.items()})
            if isinstance(widget, (ttk.Entry, ttk.Combobox)):
                widget.configure(font=("宋体", -max(11, round(self.font_size*4/3))))
        self.root.after_idle(self.wrap_info)
        self.root.after_idle(self.resize_plot)

    def wrap_info(self, event=None):
        available = max(100, self.info_group.winfo_width())
        self.status_label.configure(wraplength=available)
        self.detail_label.configure(wraplength=max(
            80, available - self.result_label.winfo_reqwidth() - round(20*self.ui_scale)))

    def resize_table(self, event):
        available = max(1, event.width - 4)
        columns = self.table["columns"]
        for column in columns:
            self.table.column(column, width=available // len(columns),
                              minwidth=max(30, round(40*self.ui_scale)))

    def resize_plot(self, event=None):
        # Matplotlib 字号单位是点，按画布 DPI 换算为与控件一致的像素大小。
        width = max(1, self.canvas.get_tk_widget().winfo_width())
        height = max(1, self.canvas.get_tk_widget().winfo_height())
        size = 12 * self.ui_scale * 72 / self.figure.dpi
        for ax in self.axes:
            ax.yaxis.label.set_size(size)
            ax.tick_params(labelsize=size * 0.9)
        self.axes[-1].xaxis.label.set_size(size)
        self.figure.subplots_adjust(
            left=min(0.22, 76*self.ui_scale/width), right=0.98,
            top=0.96, bottom=min(0.30, 42*self.ui_scale/height), hspace=0.20)
        self.canvas.draw_idle()


    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def exit_fullscreen(self, event=None):
        self.fullscreen = False
        self.root.attributes("-fullscreen", False)

    def choose_model(self):
        path = filedialog.askdirectory(title="选择模型文件夹")
        if path:
            self.model_path.set(path)

    def choose_data(self):
        path = filedialog.askopenfilename(title="选择 OpenBMI MAT 文件", filetypes=[("MAT 数据", "*.mat")])
        if path:
            self.data_path.set(path)

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        settings = dict(model=self.model_path.get(), data=self.data_path.get(),
                        mode=self.mode.get(), phase=self.phase.get(), speed=self.speed.get(),
                        host=self.host.get(), port=self.port.get())
        if settings["mode"] == "本地文件回放" and not settings["data"]:
            self.status.set("请先选择 OpenBMI MAT 文件。")
            return
        self.stop_event.clear()
        self.correct = self.count = 0
        for row in self.table.get_children():
            self.table.delete(row)
        self.result.set("正在准备")
        self.detail.set("首次加载模型和滤波需要时间，请稍候。")
        self.start_button.configure(state="disabled")
        self.worker = threading.Thread(target=self.run, args=(settings,), daemon=True)
        self.worker.start()

    def run(self, settings):
        try:
            self.events.put(("status", "正在加载模型……"))
            processor = BCIProcessor(settings["model"])
            self.events.put(("info", f"{len(processor.channels)} 通道 · 每试次 {processor.n_times} 点 / 4 秒 · 运算设备：{processor.device}"))
            if self.stop_event.is_set():
                return
            if settings["mode"] == "本地文件回放":
                self.events.put(("status", "正在读取并滤波，请稍候……"))
                items = load_trials(settings["data"], settings["phase"])
            else:
                self.events.put(("status", "正在连接发送端……"))
                self.connection = socket.create_connection((settings["host"], int(settings["port"])), timeout=10)
                self.connection.settimeout(None)
                self.events.put(("status", "已连接，等待发送试次……"))
                items = iter(lambda: receive_trial(self.connection), None)
            for index, total, trial, truth, channels in items:
                if self.stop_event.is_set():
                    break
                started = time.perf_counter()
                prediction, confidence = processor.predict(trial, channels)
                elapsed = (time.perf_counter() - started) * 1000
                self.events.put(("trial", (index, total, truth, prediction, confidence, elapsed,
                                          processor.align(trial, channels), processor.channels)))
                if settings["mode"] == "本地文件回放":
                    if self.stop_event.wait(4 / float(settings["speed"])):
                        break
            self.events.put(("status", "回放已停止。" if self.stop_event.is_set() else "回放完成。"))
        except Exception as exc:
            self.events.put(("status", "回放已停止。" if self.stop_event.is_set() else f"无法继续：{exc}"))
        finally:
            if self.connection:
                self.connection.close()
                self.connection = None
            self.events.put(("done", None))

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "info":
                    self.model_info.set(payload)
                elif kind == "done":
                    self.start_button.configure(state="normal")
                elif kind == "trial":
                    self.display_trial(payload)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def display_trial(self, item):
        index, total, truth, prediction, confidence, elapsed, trial, channels = item
        self.count += 1
        self.correct += int(truth == prediction)
        self.status.set(f"正在回放：第 {index} / {total} 个试次")
        self.result.set(f"识别动作：{LABELS[prediction]}")
        self.detail.set(f"真实动作：{LABELS[truth]}    置信度：{confidence:.1%}    "
                        f"当前符合率：{self.correct / self.count:.1%}（{self.count} 个试次）")
        row = self.table.insert("", "end", values=(f"{index}/{total}", LABELS[truth], LABELS[prediction],
                                                   f"{confidence:.1%}", f"{elapsed:.0f} ms"))
        if len(self.table.get_children()) > 200:
            self.table.delete(self.table.get_children()[0])
        self.table.see(row)
        
        mapping = {name.lower(): i for i, name in enumerate(channels)}
        for ax, name in zip(self.axes, ["C3", "Cz", "C4"]):
            ax.clear()
            ax.set_facecolor("#ffffff")
            if name.lower() in mapping:
                signal = trial[mapping[name.lower()]] * 1e6
                ax.plot(np.arange(len(signal)) / 250, signal, color="#25857c", linewidth=1.2)
            ax.set_ylabel(name + " (µV)", fontname="宋体", fontsize=9, color="#24464d")
            ax.tick_params(colors="#24464d", labelsize=8)
            ax.grid(alpha=0.15, linestyle="--", color="#688188")
            for spine in ax.spines.values():
                spine.set_color("#d0dfe2")
        self.axes[-1].set_xlabel("Time (s)", fontname="SimSun", fontsize=9, color="#24464d")
        self.resize_plot()

    def stop(self):
        self.stop_event.set()
        self.status.set("正在停止...")
        if self.connection:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def close(self):
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    OpenBMIApp(root)
    root.mainloop()
