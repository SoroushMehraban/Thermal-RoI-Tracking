import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import pandas as pd
import threading
import numpy as np
import torch
import sys
import os
import fnv.file
import mediapy as media
from tapnet.utils import transforms
from roi_utils import extract_roi_values, draw_roi_plot
from model_utils import inference, convert_points_to_query_points, model, device

class App:
    def __init__(self, root, batch_size):
        self.root = root
        self.batch_size = batch_size
        self.root.title("Thermal RoI Tracking")
        self.root.configure(bg='#0C0C0C')
        x = (self.root.winfo_screenwidth() // 2) - ((400) // 2)
        y = (self.root.winfo_screenheight() // 3) - ((200) // 2)
        self.root.geometry(f"500x200+{x}+{y}")

        self.file_button = self.file_button_init()
        self.progress_bar = self.progress_bar_init()
        self.progress_label = self.progress_label_init()
        self.error_label = self.error_label_init()

    def file_button_init(self):
        file_button = tk.Button(self.root, text="Select File", command=self.open_file,
                                width=20, height=2, bg='#F2613F', fg='white',
                                font=('Helvetica', 12, 'bold'), bd=3, relief='raised')
        file_button.grid(row=0, column=0, padx=100, pady=(100, 0))
        return file_button
    
    def progress_bar_init(self):
        progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        progress_bar.grid(row=2, column=0, padx=100, pady=(100, 10))
        progress_bar.grid_remove()  # Initially hidden

        return progress_bar
    
    def progress_label_init(self):
        progress_label = tk.Label(self.root, text="")
        progress_label.grid(row=3, column=0, padx=10, pady=(0, 10))
        progress_label.grid_remove()  # Initially hidden
        
        return progress_label

    def error_label_init(self):
        error_label = tk.Label(self.root, text="")
        error_label.grid(row=3, column=0, padx=10, pady=(10, 10))
        error_label.grid_remove()  # Initially hidden
        
        return error_label

    def read_frames(self, filepath):
        video = fnv.file.ImagerFile(filepath)
        video.unit = fnv.Unit.TEMPERATURE_FACTORY
        height = video.height
        width = video.width
        n_frames = video.num_frames
        self.video_data = np.zeros((n_frames, height, width))

        for i in range(n_frames):
            self.progress_bar['value'] = (i + 1) / n_frames * 100
            self.progress_label.config(text=f"Reading frame {i + 1}/{n_frames}",
                                       bg=self.root.cget('bg'),
                                       fg='white')
            self.root.update_idletasks()  # Update the GUI to reflect progress

            video.get_frame(i)
            self.video_data[i] = np.array(video.final, copy=False).reshape((height, width))
        
        self.progress_label.grid_remove()
        self.progress_bar.grid_remove()
        self.show_video_first_frame()
    
    def normalize_frame(self, frame):
        if not hasattr(self, 'min_video_data'):
            self.min_video_data = np.min(self.video_data)
        if not hasattr(self, 'max_video_data'):
            self.max_video_data = np.max(self.video_data)

        normalized_frame = (frame - self.min_video_data) /\
                                (self.max_video_data - self.min_video_data) * 255
        normalized_frame = normalized_frame.to(torch.uint8)

        normalized_frame = torch.stack([normalized_frame] * 3, dim=-1)
        return normalized_frame

    def show_video_first_frame(self):
        # Buttons
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_oval,
                                 bg='#F2613F', fg='white', font=('Helvetica', 12, 'bold'),
                                 bd=3, relief='raised')
        self.clear_button.grid(row=1, column=0, padx=100, sticky="ew")

        self.process_button = tk.Button(self.root, text="Process", command=self.process_oval,
                                   bg='#014421', fg='white', font=('Helvetica', 12, 'bold'),
                                   bd=3, relief='raised')
        self.process_button.grid(row=2, column=0, padx=100, sticky="ew")

        # First frame display
        first_frame = self.normalize_frame(torch.tensor(self.video_data[0]))
        first_frame = Image.fromarray(first_frame.numpy())
        self.width, self.height = first_frame.size
        visualization_width = 800
        self.scale_factor = int(np.ceil(visualization_width / self.width)) if visualization_width / self.width < 2 else int(visualization_width / self.width)
        visualization_width = self.width * self.scale_factor
        visualization_height = self.height * self.scale_factor
        first_frame = first_frame.resize((visualization_width, visualization_height))
        self.photo = ImageTk.PhotoImage(image=first_frame)

        # Canvas creation
        self.canvas = tk.Canvas(self.root, width=visualization_width, height=visualization_height)
        self.canvas.grid(row=0, column=0, padx=100, pady=(100, 0))
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.current_oval = None
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_button_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_button_release)

        # Update window size
        self.root.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        button_heights = self.clear_button.winfo_height() + self.process_button.winfo_height()

        x = (self.root.winfo_screenwidth() // 2) - ((canvas_width+200) // 2)
        y = (self.root.winfo_screenheight() // 2) - ((canvas_height + button_heights + 200) // 2)
        self.root.geometry(f"{canvas_width+200}x{canvas_height + button_heights + 200}+{x}+{y}")
    
    def clear_oval(self):
        if self.current_oval:
            self.canvas.delete(self.current_oval)
            self.current_oval = None
    
    def track_points(self, POINTS):
        resize_height, resize_width = 256, 256
        query_points = convert_points_to_query_points(0, POINTS, self.scale_factor,
                                                        self.height, self.width,
                                                        resize_height, resize_width)
        query_points = torch.tensor(query_points).to(device)

        predictions = []
        tracked_ovals = []
        for i in range(0, self.video_data.shape[0], self.batch_size):
            frames = media.resize_video(self.video_data[i:i+self.batch_size], (resize_height, resize_width))
            frames = torch.tensor(frames).to(device)
            if i == 0:
                first_frame = frames[0].unsqueeze(0)
            else:
                frames = torch.cat([first_frame, frames], dim=0)
            tracks, visibles = inference(self.normalize_frame(frames), query_points)

            POINTS, BATCH_FRAMES, XY = 0, 1, 2
            tracks = torch.squeeze(tracks).permute(BATCH_FRAMES, POINTS, XY).cpu().numpy()
            visibles = torch.squeeze(visibles).permute(BATCH_FRAMES, POINTS).cpu().numpy()
            
            tracks = transforms.convert_grid_coordinates(tracks,
                                                        (resize_width, resize_height),
                                                        (self.width, self.height))
            if i > 0:
                tracks = tracks[1:]
                visibles = visibles[1:]
            predictions.append({'tracks':tracks, 'visibles':visibles})

            # Update GUI
            if i == 0:
                self.canvas.grid()
                self.canvas.config(width=self.video_data.shape[2],
                                   height=self.video_data.shape[1])
                self.progress_bar.grid_configure(pady=(10, 10))

                canvas_width = self.video_data.shape[2]
                canvas_height = self.video_data.shape[1]

                x = (self.root.winfo_screenwidth() // 2) - ((canvas_width+200) // 2)
                y = (self.root.winfo_screenheight() // 3) - ((canvas_height + 200) // 2)
                self.root.geometry(f"{canvas_width+200}x{canvas_height + 200}+{x}+{y}")

            effective_batch_size = tracks.shape[0] -1 if i > 0 else tracks.shape[0]
            last_frame_batch = torch.tensor(self.video_data[i + effective_batch_size - 1])
            self.photo = ImageTk.PhotoImage(Image.fromarray(self.normalize_frame(last_frame_batch).numpy()))
            self.canvas.itemconfig(self.image_on_canvas, image=self.photo)
            
            if self.current_oval is not None:
                self.canvas.delete(self.current_oval)
            if len(tracked_ovals) > 0:
                for tracked_oval in tracked_ovals:
                    self.canvas.delete(tracked_oval)
                tracked_ovals = []

            min_x, min_y = np.min(tracks[-1], axis=0)
            max_x, max_y = np.max(tracks[-1], axis=0)
            self.current_oval = self.canvas.create_oval(min_x, min_y, max_x, max_y, outline='red')
            for (x, y) in tracks[-1]:
                tracked_oval = self.canvas.create_oval(x-2, y-2, x+2, y+2, outline='blue', fill='blue')
                tracked_ovals.append(tracked_oval)

            self.progress_bar['value'] = (i + 1) / self.video_data.shape[0] * 100
            self.progress_label.config(text=f"Processed frames {i + 1}/{self.video_data.shape[0]}",
                                       bg=self.root.cget('bg'),
                                       fg='white')

        self.tracks = np.concatenate([x['tracks'] for x in predictions])
        self.visibles = np.concatenate([x['visibles'] for x in predictions])

        print(self.tracks.shape, self.visibles.shape)
        
        self.progress_bar.grid_remove()
        self.progress_label.grid_remove()

        self.show_final_buttons()

    def save_roi(self):
        folder_selected = filedialog.askdirectory(title="Select Folder to Save RoI")
        if folder_selected:
            self.in_progress.grid()

            if not hasattr(self, 'roi'):
                self.roi = extract_roi_values(self.video_data, self.tracks)
                self.visibles = np.sum(self.visibles, axis=1) > 2
            frames = [i for i in range(self.video_data.shape[0])]
            df = pd.DataFrame({
                'Frame': frames,
                'RoI': self.roi,
                'Visible': self.visibles
            })
            file_name = os.path.join(folder_selected, 'out.xlsx')
            count = 1
            while True:
                if os.path.exists(file_name):
                    count += 1
                    file_name = os.path.join(folder_selected, f'out{count}.xlsx')
                else:
                    break
            df.to_excel(file_name, index=False)

            self.in_progress.grid_remove()
            self.button_save_roi.configure(bg='#5ced73', fg='black', text='RoI saved', state='disabled')
    
    def save_tracks(self):
        folder_selected = filedialog.askdirectory(title="Select Folder to Save tracks")
        if folder_selected:
            self.in_progress.grid()

            file_name = os.path.join(folder_selected, 'tracks.npy')
            count = 1
            while True:
                if os.path.exists(file_name):
                    count += 1
                    file_name = os.path.join(folder_selected, f'tracks{count}.npy')
                else:
                    break

            np.save(file_name, self.tracks)

            self.in_progress.grid_remove()
            self.button_save_tracks.configure(bg='#5ced73', fg='black', text='Tracks saved', state='disabled')

    def save_scatter(self):
        folder_selected = filedialog.askdirectory(title="Select Folder to Save scatters")
        if folder_selected:
            self.in_progress.grid()

            if not hasattr(self, 'roi'):
                self.roi = extract_roi_values(self.video_data, self.tracks)
                self.visibles = np.sum(self.visibles, axis=1) > 2
            frames = np.array([i for i in range(self.video_data.shape[0])])

            file_name = os.path.join(folder_selected, 'roi.png')
            count = 1
            while True:
                if os.path.exists(file_name):
                    count += 1
                    file_name = os.path.join(folder_selected, f'roi{count}.png')
                else:
                    break
            
            draw_roi_plot(frames, self.roi, self.visibles, file_name)
            self.in_progress.grid_remove()
            self.button_save_scatter.configure(bg='#5ced73', fg='black', text='Scatter saved', state='disabled')

    def start_again(self):
        self.root.destroy()

    def show_final_buttons(self):
        self.canvas.grid_configure(padx=(100, 0), pady=(100, 0), rowspan=4)
        self.button_save_roi = tk.Button(self.root, text="Save RoI", command=self.save_roi,
                                         bg='#F2613F', fg='white',
                                         font=('Helvetica', 12, 'bold'), bd=3, relief='raised')
        self.button_save_roi.grid(row=0, column=1, pady=(100, 0), sticky='nsew')

        self.button_save_tracks = tk.Button(self.root, text="Save Tracks", command=self.save_tracks,
                                            bg='#F2613F', fg='white',
                                            font=('Helvetica', 12, 'bold'), bd=3, relief='raised')
        self.button_save_tracks.grid(row=1, column=1, sticky='nsew')

        self.button_save_scatter = tk.Button(self.root, text="Save Scatter Figure", command=self.save_scatter,
                                             bg='#F2613F', fg='white',
                                             font=('Helvetica', 12, 'bold'), bd=3, relief='raised')
        self.button_save_scatter.grid(row=2, column=1, sticky='new')

        self.button_start_again = tk.Button(self.root, text="Start again", command=self.start_again,
                                            font=('Helvetica', 12, 'bold'), bd=3, relief='raised')
        self.button_start_again.grid(row=3, column=1, sticky='sew')

        self.in_progress = tk.Label(self.root, text="In progress ...", bg=self.root.cget('bg'),
                                   fg='#5ced73', font=('Helvetica', 12, 'bold'))
        self.in_progress.grid(row=4, column=0, pady=10)
        self.in_progress.grid_remove()  # Initially hidden

        # Update window size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - ((canvas_width+400) // 2)
        y = (self.root.winfo_screenheight() // 2) - ((canvas_height + 200) // 2)
        self.root.geometry(f"{canvas_width+400}x{canvas_height + 200}+{x}+{y}")

    def process_oval(self):
        if self.current_oval:
            x0, y0, x1, y1 = self.canvas.coords(self.current_oval)

            points = [(y0, (x0 + x1) / 2), (y1, (x0 + x1) / 2),
                      ((y0 + y1) / 2, x0), ((y0 + y1) / 2, x1)]

            thread = threading.Thread(target=self.track_points, args=(points,))
            thread.daemon = True
            thread.start()

            self.canvas.grid_remove()
            self.process_button.grid_remove()
            self.clear_button.grid_remove()

            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            
            self.progress_bar.grid()
            self.progress_label.grid()
            self.progress_bar['value'] = 0
            self.progress_label.config(text=f"Processed frames 0/{self.video_data.shape[0]}",
                                        bg=self.root.cget('bg'),
                                        fg='white')
            x = (self.root.winfo_screenwidth() // 2) - ((500) // 2)
            y = (self.root.winfo_screenheight() // 3) - ((200) // 2)
            self.root.geometry(f"500x200+{x}+{y}")

    def on_canvas_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.current_oval:
            self.canvas.delete(self.current_oval)
            self.current_oval = None

    def on_canvas_mouse_drag(self, event):
        if self.current_oval:
            self.canvas.delete(self.current_oval)
        self.current_oval = self.canvas.create_oval(self.start_x, self.start_y, event.x, event.y, outline='red')

    def on_canvas_button_release(self, event):
        if self.current_oval:
            self.canvas.delete(self.current_oval)
        self.current_oval = self.canvas.create_oval(self.start_x, self.start_y, event.x, event.y, outline='red')

    def open_file(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            if filepath.split('.')[-1] == 'seq':
                thread = threading.Thread(target=self.read_frames, args=(filepath,))
                thread.daemon = True
                thread.start()

                self.progress_bar.grid()
                self.error_label.grid_remove()
                self.progress_label.grid()
                self.file_button.grid_remove()
                self.progress_bar['value'] = 0
            else:
                self.error_label.grid()
                self.error_label.config(text="Selected file should be a .seq file",
                                   bg=self.root.cget('bg'),
                                   fg='red')

    def on_closing(self):
        self.root.destroy()
        sys.exit()
        
