import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import cv2  # Для работы с видео
import time
class ImageApp4:
    def __init__(self, root):
        # Инициализация
        self.root = root
        self.root.title("Image and Video App")

        # Списки для хранения изображений, видео и их признаков
        self.images = []
        self.features = []
        self.video_capture = None
        self.video_label = None
        self.video_path = None
        self.playing = False 
        self.playing_with_sub= False  # Флаг для отслеживания состояния воспроизведения видео
        self.fps = 30  
        self.bg_subtraction_enabled = False
         # Создаем фреймы для левой и правой частей
        self.left_frame = tk.Frame(root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.right_frame = tk.Frame(root)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ne")

        # ЛЕВАЯ ЧАСТЬ (Изображения и кнопки)
        self.image_frame = tk.Frame(self.left_frame)
        self.image_frame.grid(row=1, pady=10)

        # Кнопки для загрузки, сравнения и очистки изображений
        self.button_frame = tk.Frame(self.left_frame)
        self.button_frame.grid(row=0, pady=10)

        self.load_button = tk.Button(self.button_frame, text="Загрузить изображения",width=20, command=self.load_images)
        self.load_button.grid(column=0, row=0, padx=2)

        self.compare_button = tk.Button(self.button_frame, text="Найти похожие",width=20, command=self.compare_images)
        self.clear_button = tk.Button(self.button_frame, text="Скрыть изображения",width=20, command=self.clear_images)

        self.result_frame = tk.Frame(self.left_frame)
        self.result_frame.grid(pady=10)
        # ПРАВАЯ ЧАСТЬ (Видео и кнопки)
        self.video_frame = tk.Frame(self.right_frame)
        self.video_frame.grid(row=1, pady=10)
        self.video_button_frame = tk.Frame(self.right_frame)
        self.video_button_frame.grid(row=0, pady=10)

        self.load_video_button = tk.Button(self.video_button_frame, text="Загрузить видео",width=20, command=self.load_video)
        self.load_video_button.grid(column=0, row=0, padx=2)

        self.clear_video_button = tk.Button(self.video_button_frame, text="Убрать видео",width=20, command=self.clear_video)
        self.bg_replace_button = tk.Button(self.video_button_frame, text="Заменить фон",width=20, command=self.replace_background)
        self.reset_button = tk.Button(self.video_button_frame, text="Сбросить фон",width=20, command=self.reset_background)
        self.flow_button = tk.Button(self.video_button_frame, text="Оптический поток",width=20, command=self.toggle_optical_flow)
        self.blur_strength = tk.DoubleVar(value=0)
        self.blur_slider = tk.Scale(self.video_button_frame, from_=0, to=20,variable=self.blur_strength, orient='horizontal', label="Размытие")

        self.simple_bg_sub_button = tk.Button(self.video_button_frame, text="Вычесть фон", width=20, command=self.toggle_bg_subtraction)
        self.simple_bg_sub_button.grid(column=5, row=0, padx=2)
      
        

        # Модель ResNet18
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))  # Удаляем последний слой классификации

        self.frame_size = (640, 360)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.background_image = None

        self.optical_flow_enabled = False
        self.prev_gray_frame = None
        self.start_time = None  
        
        # Инициализация вычитателя фона
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.background_image = None  # Статичное изображение для замены фона

       
        
        self.optical_flow_enabled = False
        self.prev_gray_frame = None  # Для хранения предыдущего кадра для расчета оптического потока
        self.start_time = None

        self.compare_button.grid_remove()
        self.clear_button.grid_remove()
        self.clear_video_button.grid_remove()
        self.bg_replace_button.grid_remove()
        self.reset_button.grid_remove()
        self.flow_button.grid_remove()
        self.blur_slider.grid_remove()
        self.result_frame.grid_remove()
        self.image_frame.grid_remove()
        self.simple_bg_sub_button.grid_remove()
    # Функция для извлечения признаков изображения
    def extract_features(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = preprocess(image).unsqueeze(0)  # Добавляем размерность пакета
        with torch.no_grad():  # Отключаем градиенты
            features = self.model(img_tensor).squeeze().numpy()  # Получаем признаки
        return features

    # Функция для загрузки изображений
    def load_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")], title="Выберите изображения")
        
        # Проверяем, что общее количество изображений не превысит 10
        if len(self.images) + len(file_paths) > 10:
            messagebox.showerror("Ошибка", "Нельзя загружать более 10 изображений.")
            return

        for file in file_paths:
            img = Image.open(file)
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)  # Изменяем размер изображения с сохранением пропорций

            self.images.append((img, file))
            img_features = self.extract_features(img)  # Извлекаем признаки для каждого изображения
            self.features.append(img_features)

        self.image_frame.grid(row=1, pady=10)
        self.compare_button.grid(column=1, row=0, padx=2)
        self.clear_button.grid(column=2, row=0, padx=2)
        self.display_images()

    # Функция для очистки всех изображений и их признаков
    def clear_images(self):
        self.images.clear()
        self.features.clear()
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        self.compare_button.grid_remove()
        self.clear_button.grid_remove()
        self.image_frame.grid_remove()
        self.result_frame.grid_remove()
    # Функция для вычисления сходства изображений (евклидово расстояние между признаками)
    def compare_images(self):
        if len(self.images) < 3:
            messagebox.showerror("Ошибка", "Загрузите минимум 3 изображения.")
            return

        # Рассчитываем евклидовы расстояния между всеми парами изображений
        dist_matrix = euclidean_distances(self.features)

        # Находим пару изображений с наименьшим расстоянием (максимально похожие)
        indices = np.unravel_index(np.argmin(dist_matrix + np.eye(len(dist_matrix)) * np.max(dist_matrix)), dist_matrix.shape)
        most_similar_pair = indices
        
        # Выводим два наиболее похожих изображения
        self.display_similar_images(most_similar_pair)

    # Функция для отображения изображений в GUI
    def display_images(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        for i, (img, file) in enumerate(self.images):
            img_tk = ImageTk.PhotoImage(img)
            label = tk.Label(self.image_frame, image=img_tk)
            label.image = img_tk
            label.grid(row=i // 4, column=i % 4, padx=2, pady=5)  # Рисуем 5 изображений в ряду

    # Функция для отображения двух наиболее похожих изображений
    def display_similar_images(self, pair):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        img1, img2 = self.images[pair[0]], self.images[pair[1]]
        img1_tk = ImageTk.PhotoImage(img1[0])
        img2_tk = ImageTk.PhotoImage(img2[0])

        self.result_frame.grid(pady=10)
        # Отображаем два изображения горизонтально
        label1 = tk.Label(self.result_frame, image=img1_tk)
        label1.image = img1_tk
        label1.grid(column=0, row=0, padx=2, pady=5)

        label2 = tk.Label(self.result_frame, image=img2_tk)
        label2.image = img2_tk
        label2.grid(column=1, row=0, padx=2, pady=5)
        
  
        # Функция для загрузки и отображения видео
    def reset_effects(self):
    
        
        # Сброс оптического потока
        self.optical_flow_enabled = False
        self.flow_button.config(text="Оптический поток")
        self.prev_gray_frame = None  # Очищаем предыдущий кадр

        # Сброс вычитания фона
        self.bg_subtraction_enabled = False
        self.prev_fg_mask = None  # Сбрасываем предыдущую маску
        self.simple_bg_sub_button.config(text="Вычесть фон")

        # Сброс фонового изображения
        self.background_image = None
        
    def load_video(self):
        self.reset_effects()
        self.clear_video()  # Очистка предыдущего видео
        self.blur_strength.set(0)
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")], title="Выберите видео")
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            print(f"FPS: {self.fps}")  # Отладочная информация
            self.delay = max(int(1000 / self.fps), 1)
            
            # Сброс переменных перед воспроизведением нового видео
            self.start_time = None
            self.prev_gray_frame = None  
            
            self.video_label = tk.Label(self.video_frame)
            self.video_label.grid(column=0, row=1, columnspan=5)
            
            self.playing = True
            
            self.clear_video_button.grid(column=1, row=0, padx=2)
            self.bg_replace_button.grid(column=3, row=0, padx=2)
            self.reset_button.grid(column=4, row=0, padx=2)
            self.flow_button.grid(column=2, row=0, padx=2)
            self.blur_slider.grid(column=0, row=1, padx=2)
            self.video_frame.grid(row=1, pady=10)
            self.simple_bg_sub_button.grid(column=5, row=0, padx=2)
            self.play_video()
    def reset_background(self):
        self.background_image = None

    def toggle_bg_subtraction(self):
    
        self.bg_subtraction_enabled = not self.bg_subtraction_enabled  
        
        if self.bg_subtraction_enabled:
            self.playing_with_sub= True
            self.simple_bg_sub_button.config(text="Вернуть обычное видео")
            self.start_time = None  
        else:
            self.playing_with_sub= False
            self.simple_bg_sub_button.config(text="Вычесть фон")
            self.start_time = None  

    
    def play_video(self):
        if self.playing:
            if self.start_time is None:
                self.start_time = time.time()
            ret, frame = self.video_capture.read()

            if ret:
                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Проверка, активирован ли режим отображения только вычитания фона
                if self.bg_subtraction_enabled:
                    fg_mask = self.bg_subtractor.apply(frame)

                    # Убедимся, что маска является двоичной
                    _, fg_mask_bin = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

                    # Создаем инверсную маску для фона
                    bg_mask_bin = cv2.bitwise_not(fg_mask_bin)

                    # Оставляем цвета переднего плана, используя fg_mask_bin
                    foreground_colored = cv2.bitwise_and(frame_rgb, frame_rgb, mask=fg_mask_bin)

                    # Задний план делаем черным
                    background_black = np.zeros_like(frame_rgb)

                    # Соединяем передний план с черным фоном
                    combined_frame = cv2.add(foreground_colored, background_black)

                    frame_image = Image.fromarray(combined_frame)
                else:
                    # Оригинальная обработка кадра (как в вашем коде)
                    if self.optical_flow_enabled:
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        small_gray_frame = cv2.resize(gray_frame, (gray_frame.shape[1] // 2, gray_frame.shape[0] // 2))

                        if self.prev_gray_frame is not None:
                            small_prev_gray = cv2.resize(self.prev_gray_frame, (self.prev_gray_frame.shape[1] // 2, self.prev_gray_frame.shape[0] // 2))
                            flow = cv2.calcOpticalFlowFarneback(
                                small_prev_gray, small_gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                            )
                            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            hsv = np.zeros_like(cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)))
                            hsv[..., 1] = 255
                            hsv[..., 0] = ang * 180 / np.pi / 2
                            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                            flow_rgb_large = cv2.resize(flow_rgb, (frame.shape[1], frame.shape[0]))
                            frame_rgb = cv2.addWeighted(frame_rgb, 0.6, flow_rgb_large, 0.4, 0)

                        self.prev_gray_frame = gray_frame

                    fg_mask = self.bg_subtractor.apply(frame)
                    blur_value = int(self.blur_strength.get())
                    if blur_value > 0:
                        _, fg_mask_bin = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                        blurred_frame = cv2.GaussianBlur(frame_rgb, (blur_value*2+1, blur_value*2+1), 0)
                        fg_mask_colored = cv2.cvtColor(fg_mask_bin, cv2.COLOR_GRAY2BGR)
                        bg_mask_colored = cv2.bitwise_not(fg_mask_colored)
                        foreground = cv2.bitwise_and(blurred_frame, fg_mask_colored)
                        background = cv2.bitwise_and(frame_rgb, bg_mask_colored)
                        frame_rgb = cv2.add(foreground, background)

                    if self.background_image is not None:
                        _, fg_mask_bin = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                        bg_mask = cv2.bitwise_not(fg_mask_bin)
                        fg_mask_colored = cv2.cvtColor(fg_mask_bin, cv2.COLOR_GRAY2BGR)
                        bg_mask_colored = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)
                        foreground = cv2.bitwise_and(frame_rgb, fg_mask_colored)
                        background = cv2.bitwise_and(self.background_image, bg_mask_colored)
                        combined_frame = cv2.add(foreground, background)
                        frame_image = Image.fromarray(combined_frame)
                    else:
                        frame_image = Image.fromarray(frame_rgb)

                frame_image = ImageTk.PhotoImage(frame_image)
                self.video_label.config(image=frame_image)
                self.video_label.image = frame_image

                elapsed_time = (time.time() - self.start_time) * 1000
                frame_index = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                expected_frame_time = (frame_index / self.fps) * 1000
                delay = max(int(expected_frame_time - elapsed_time), 1)

                self.root.after(delay, self.play_video)
            else:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.start_time = None
                self.root.after(1, self.play_video)
                
                
    def toggle_optical_flow(self):
        self.optical_flow_enabled = not self.optical_flow_enabled
        if self.optical_flow_enabled:
            self.flow_button.config(text="Отключить поток")
        else:
            self.flow_button.config(text="Оптический поток")
        self.prev_gray_frame = None  # Сбрасываем предыдущий кадр
    def replace_background(self):
        
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")], title="Выберите изображение для замены фона")
        if image_path:
            bg_image = cv2.imread(image_path)
            bg_image = cv2.resize(bg_image, self.frame_size)  # Изменяем размер изображения под размер кадра
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB
            self.background_image = bg_image
    # Функция для очистки видео
    def clear_video(self):
        if self.video_capture is not None:
            self.video_capture.release()  # Освобождаем ресурсы видеозахвата
            self.video_capture = None
            self.playing = False 
            self.playing_with_sub= False # Останавливаем воспроизведение
            

        children = self.video_frame.winfo_children()
        if children:  # Проверяем, что есть хотя бы один виджет
            children[0].destroy() # Удаляем все виджеты из видео фрейма
        self.clear_video_button.grid_remove()
        self.bg_replace_button.grid_remove()
        self.reset_button.grid_remove()
        self.video_frame.grid_remove()
        self.flow_button.grid_remove()
        self.simple_bg_sub_button.grid_remove()
        self.reset_effects()
        if self.blur_slider.winfo_exists():
            self.blur_slider.grid_remove()

# Создание основного окна приложения
root = tk.Tk()
app = ImageApp4(root)
root.mainloop()
