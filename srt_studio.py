import os
import sys
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "SRT Translation Studio"
TRANSLATOR_SCRIPT = "translator.py"
COMPARATOR_SCRIPT = "compare_srt.py"


class SRTStudio(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.process = None
        self.output_queue = queue.Queue()

        self.input_dir = tk.StringVar(value="input")
        self.output_dir = tk.StringVar(value="output")
        self.compare_dir = tk.StringVar(value="comparison")

        self.status = tk.StringVar(value="พร้อมใช้งาน")

        self._build_style()
        self._build_ui()
        self.after(100, self._poll_output)

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _build_style(self):
        style = ttk.Style(self)

        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 20, "bold")
        )

        style.configure(
            "Section.TLabelframe.Label",
            font=("Segoe UI", 11, "bold")
        )

        style.configure(
            "Big.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(18, 10)
        )

    def _build_ui(self):
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)

        # Header
        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 15))

        ttk.Label(
            header,
            text="SRT Translation Studio",
            style="Title.TLabel"
        ).pack(side="left")

        ttk.Label(
            header,
            textvariable=self.status
        ).pack(side="right", pady=5)

        # Paths
        paths = ttk.LabelFrame(
            root,
            text="โฟลเดอร์โปรเจกต์",
            style="Section.TLabelframe",
            padding=12
        )
        paths.pack(fill="x", pady=(0, 12))

        self._path_row(
            paths,
            0,
            "Input SRT",
            self.input_dir
        )

        self._path_row(
            paths,
            1,
            "Output",
            self.output_dir
        )

        self._path_row(
            paths,
            2,
            "Comparison",
            self.compare_dir
        )

        # Main actions
        actions = ttk.LabelFrame(
            root,
            text="การทำงาน",
            style="Section.TLabelframe",
            padding=15
        )
        actions.pack(fill="x", pady=(0, 12))

        translate_frame = ttk.Frame(actions)
        translate_frame.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ttk.Label(
            translate_frame,
            text="TRANSLATOR",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")

        ttk.Label(
            translate_frame,
            text="แปล SRT ด้วย translator.py / Ollama"
        ).pack(anchor="w", pady=(2, 8))

        ttk.Button(
            translate_frame,
            text="▶  เริ่มแปล SRT",
            style="Big.TButton",
            command=self.run_translator
        ).pack(fill="x")

        compare_frame = ttk.Frame(actions)
        compare_frame.pack(side="left", fill="x", expand=True, padx=(8, 0))

        ttk.Label(
            compare_frame,
            text="COMPARATOR",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w")

        ttk.Label(
            compare_frame,
            text="เปรียบเทียบ Source / Gold / RAG"
        ).pack(anchor="w", pady=(2, 8))

        ttk.Button(
            compare_frame,
            text="⇄  เปรียบเทียบ SRT",
            style="Big.TButton",
            command=self.run_comparator
        ).pack(fill="x")

        # Quick controls
        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(0, 12))

        ttk.Button(
            controls,
            text="เปิด Input",
            command=lambda: self.open_folder(self.input_dir.get())
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            controls,
            text="เปิด Output",
            command=lambda: self.open_folder(self.output_dir.get())
        ).pack(side="left", padx=6)

        ttk.Button(
            controls,
            text="เปิด Comparison",
            command=lambda: self.open_folder(self.compare_dir.get())
        ).pack(side="left", padx=6)

        ttk.Button(
            controls,
            text="ล้าง Console",
            command=self.clear_console
        ).pack(side="right")

        ttk.Button(
            controls,
            text="หยุด",
            command=self.stop_process
        ).pack(side="right", padx=(0, 6))

        # Console
        console_frame = ttk.LabelFrame(
            root,
            text="Console / Progress",
            style="Section.TLabelframe",
            padding=8
        )
        console_frame.pack(fill="both", expand=True)

        self.console = tk.Text(
            console_frame,
            wrap="word",
            font=("Consolas", 10),
            bg="#111111",
            fg="#eeeeee",
            insertbackground="#ffffff"
        )

        scrollbar = ttk.Scrollbar(
            console_frame,
            orient="vertical",
            command=self.console.yview
        )

        self.console.configure(
            yscrollcommand=scrollbar.set
        )

        self.console.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.write_console(
            "SRT Translation Studio พร้อมใช้งาน\n"
            f"Python: {sys.executable}\n"
        )

    def _path_row(self, parent, row, label, variable):
        ttk.Label(
            parent,
            text=label,
            width=14
        ).grid(
            row=row,
            column=0,
            sticky="w",
            pady=3
        )

        entry = ttk.Entry(
            parent,
            textvariable=variable
        )

        entry.grid(
            row=row,
            column=1,
            sticky="ew",
            padx=8,
            pady=3
        )

        ttk.Button(
            parent,
            text="เลือก",
            command=lambda v=variable: self.choose_folder(v)
        ).grid(
            row=row,
            column=2,
            pady=3
        )

        parent.columnconfigure(1, weight=1)

    # ---------------------------------------------------------
    # Process
    # ---------------------------------------------------------

    def _script_exists(self, script):
        if not os.path.exists(script):
            messagebox.showerror(
                "ไม่พบไฟล์",
                f"ไม่พบ {script}\n\n"
                "วาง GUI ไว้ในโฟลเดอร์เดียวกับ "
                "translator.py และ compare_srt.py"
            )
            return False

        return True

    def run_script(self, script):
        if self.process and self.process.poll() is None:
            messagebox.showwarning(
                "กำลังทำงาน",
                "มี process กำลังทำงานอยู่แล้ว"
            )
            return

        if not self._script_exists(script):
            return

        # สร้าง directory ที่โปรแกรมหลักคาดหวัง
        os.makedirs(self.input_dir.get(), exist_ok=True)
        os.makedirs(self.output_dir.get(), exist_ok=True)
        os.makedirs(self.compare_dir.get(), exist_ok=True)

        self.status.set(f"กำลังทำงาน: {script}")
        self.write_console(
            "\n"
            + "=" * 80
            + f"\nSTART: {script}\n"
            + "=" * 80
            + "\n"
        )

        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=os.getcwd()
            )

            threading.Thread(
                target=self._read_process_output,
                daemon=True
            ).start()

        except Exception as e:
            self.status.set("เกิดข้อผิดพลาด")
            messagebox.showerror(
                "ไม่สามารถเริ่มโปรแกรม",
                str(e)
            )

    def _read_process_output(self):
        process = self.process

        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    self.output_queue.put(("output", line))

            process.stdout.close()

            return_code = process.wait()

            self.output_queue.put(
                ("done", return_code)
            )

        except Exception as e:
            self.output_queue.put(
                ("output", f"\n[GUI ERROR] {e}\n")
            )

    def _poll_output(self):
        try:
            while True:
                event, data = self.output_queue.get_nowait()

                if event == "output":
                    self.write_console(data)

                elif event == "done":
                    if data == 0:
                        self.status.set("เสร็จสิ้น")
                        self.write_console(
                            "\n"
                            + "=" * 80
                            + "\nPROCESS FINISHED: SUCCESS\n"
                            + "=" * 80
                            + "\n"
                        )
                    else:
                        self.status.set(
                            f"จบด้วย error (code {data})"
                        )
                        self.write_console(
                            "\n"
                            + "=" * 80
                            + f"\nPROCESS FINISHED: ERROR {data}\n"
                            + "=" * 80
                            + "\n"
                        )

                    self.process = None

        except queue.Empty:
            pass

        self.after(100, self._poll_output)

    # ---------------------------------------------------------
    # Translator / Comparator
    # ---------------------------------------------------------

    def run_translator(self):
        self.run_script(TRANSLATOR_SCRIPT)

    def run_comparator(self):
        self.run_script(COMPARATOR_SCRIPT)

    # ---------------------------------------------------------
    # Folder helpers
    # ---------------------------------------------------------

    def choose_folder(self, variable):
        path = filedialog.askdirectory()

        if path:
            variable.set(path)

    def open_folder(self, path):
        if not path:
            return

        path = os.path.abspath(path)

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        try:
            os.startfile(path)
        except Exception as e:
            messagebox.showerror(
                "เปิดโฟลเดอร์ไม่ได้",
                str(e)
            )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    def write_console(self, text):
        self.console.insert("end", text)
        self.console.see("end")

    def clear_console(self):
        self.console.delete("1.0", "end")

    # ---------------------------------------------------------
    # Stop
    # ---------------------------------------------------------

    def stop_process(self):
        if not self.process:
            self.status.set("ไม่มี process ที่กำลังทำงาน")
            return

        if self.process.poll() is not None:
            self.process = None
            self.status.set("ไม่มี process ที่กำลังทำงาน")
            return

        answer = messagebox.askyesno(
            "หยุดการทำงาน",
            "ต้องการหยุด process ที่กำลังทำงานหรือไม่?"
        )

        if not answer:
            return

        try:
            self.process.terminate()
            self.write_console(
                "\n⚠ PROCESS ถูกสั่งหยุดโดยผู้ใช้\n"
            )
            self.status.set("หยุดการทำงานแล้ว")

        except Exception as e:
            messagebox.showerror(
                "หยุด process ไม่สำเร็จ",
                str(e)
            )


if __name__ == "__main__":
    app = SRTStudio()
    app.mainloop()
