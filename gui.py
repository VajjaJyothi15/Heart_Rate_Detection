import tkinter as tk
from PIL import Image, ImageTk
import cv2

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class MedicalGUI:

    def __init__(self, app):

        self.app = app
        self.running = False

        self.window = tk.Tk()
        self.window.title("AI HEALTH MONITOR")
        self.window.geometry("1300x800")
        self.window.configure(bg="black")

        # ================= HEADER =================
        tk.Label(
            self.window,
            text="AI HEALTH MONITOR",
            font=("Arial", 22, "bold"),
            fg="white",
            bg="black"
        ).pack()

        # ================= TOP FRAME =================
        top = tk.Frame(self.window, bg="black")
        top.pack()

        self.cam_label = tk.Label(top)
        self.cam_label.grid(row=0, column=0)

        self.eye_label = tk.Label(top)
        self.eye_label.grid(row=0, column=1)

        # ================= INFO =================
        self.info = tk.Label(
            self.window,
            text="BPM: -- | Freq: -- | Confidence: --",
            font=("Arial", 14),
            fg="lime",
            bg="black"
        )
        self.info.pack()

        # ================= GRAPH FRAME =================
        graph_frame = tk.Frame(self.window, bg="black")
        graph_frame.pack()

        # ---- Blink Graph ----
        self.fig1 = Figure(figsize=(4, 2))
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, graph_frame)
        self.canvas1.get_tk_widget().grid(row=0, column=0)

        # ---- FFT Graph ----
        self.fig2 = Figure(figsize=(4, 2))
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, graph_frame)
        self.canvas2.get_tk_widget().grid(row=0, column=1)

        # ================= BUTTONS =================
        btn = tk.Frame(self.window, bg="black")
        btn.pack()

        tk.Button(btn, text="Open", command=self.start).grid(row=0, column=0)
        tk.Button(btn, text="Stop", command=self.stop).grid(row=0, column=1)
        tk.Button(btn, text="Exit", command=self.window.destroy).grid(row=0, column=2)

        self.update()

    # ================= CONTROL =================

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    # ================= UPDATE =================

    def update(self):

        if self.running:

            frame, roi, data = self.app.get_frame()

            if frame is not None:

                # ---- Camera ----
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (500, 350))

                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.cam_label.imgtk = img
                self.cam_label.config(image=img)

                # ---- Eye Crop ----
                if roi is not None:
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi = cv2.resize(roi, (250, 150))

                    img2 = ImageTk.PhotoImage(Image.fromarray(roi))
                    self.eye_label.imgtk = img2
                    self.eye_label.config(image=img2)

                # ---- Info ----
                self.info.config(
                    text=f"Heart Rate: {data['bpm']:.2f} BPM | "
                           f"Frequency: {data['freq']:.2f} Hz | "
                           f"Confidence: {data['confidence']:.2f}%"
                )

                # ---- BLINK GRAPH ----
                self.ax1.clear()
                self.ax1.plot(self.app.blink_buffer, color="red")
                self.ax1.set_title("Blink Signal")
                self.canvas1.draw()

                # ---- FFT GRAPH ----
                if hasattr(self.app, "last_power"):
                    self.ax2.clear()
                    self.ax2.plot(self.app.last_power, color="yellow")
                    self.ax2.set_title("FFT Spectrum")
                    self.canvas2.draw()

        self.window.after(30, self.update)

    def run(self):
        self.window.mainloop()