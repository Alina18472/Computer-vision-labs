import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import re
import easyocr
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
class ImageApp3:
    def __init__(self, arg1):
        self.arg1 = arg1
        open_image_button = tk.Button(text="Открыть файл", command=self.open_image)
        open_image_button.grid(row=0,sticky=tk.W, column=0,  padx=10)
        self.current_image = None 
        self.original_image = None  
        # Фрейм для изображения
        self.image_frame = tk.Frame(self.arg1)
        self.image_frame.grid(row=1, column=3, sticky=tk.W+tk.N)
        self.image_label = tk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, padx=10, sticky=tk.N + tk.EW)
        self.segment_label = tk.Label(self.image_frame)
        self.segment_label.grid(row=1, column=0, padx=10, sticky=tk.N + tk.EW)
        self.button_frame = tk.Frame()
        self.remove_button= None
        
        
    def create_interface(self):
        
        self.button_frame.grid(row=1,column=0,sticky=tk.N,pady=10)
        self.reset_button = tk.Button(self.button_frame,text="Восстановить изображение", command=self.reset)
        self.reset_button.grid(row=0,padx=10, sticky=tk.N + tk.EW)
        self.canny_button = tk.Button(self.button_frame,text="Детектор Canny", command=self.canny_edge_detection)
        self.canny_button.grid(row=1, padx=10, sticky=tk.N + tk.EW)
        self.roberts_button = tk.Button(self.button_frame, text="Оператор Робертса", command=self.roberts_edge_detection)
        self.roberts_button.grid(row=2,padx=10, sticky=tk.N + tk.EW)
        
        self.keypoints_button = tk.Button(self.button_frame, text="Ключевые точки", command=self.detect_keypoints)
        self.keypoints_button.grid(row=3, padx=10, sticky=tk.N + tk.EW)

        # Кнопка для выделения номерного знака
        self.license_plate_button = tk.Button(self.button_frame, text="Выделить номерной знак", command=self.detect_license_plate)
        self.license_plate_button.grid(row=4, padx=10, sticky=tk.N + tk.EW)

        tk.Label(self.button_frame, text="Введите количество кластеров (от 2 до 20):").grid(row=5, padx=10,sticky=tk.W)
        self.cluster_entry = tk.Entry(self.button_frame)
        
        self.cluster_entry.grid(row=6, column=0, sticky=tk.W, padx=15)

        self.segmentation_button = tk.Button(self.button_frame, text="Сегментация", command=self.segment_image)
        self.segmentation_button.grid(row=7, padx=10, sticky=tk.N + tk.EW)
        
    
    def reset(self):
        self.show_image(self.original_image, self.image_label)
        # self.current_image = self.original_image.copy() 
   

    def remove_image(self):
        self.segment_label.config(image='') 
        if self.remove_button is not None:
            self.remove_button.grid_forget()
            self.remove_button=None

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JPEG", "*.jpg;*.jpeg"),("PNG", "*.png"),("GIF", "*.gif"),("Все файлы", "*.*"),])
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                self.current_image = self.original_image.copy()  # Сохраняем оригинал при открытии
                self.show_image(self.original_image,self.image_label)
                self.segment_label.config(image='')
                self.create_interface() 
                if self.remove_button is not None:
                    self.remove_button.grid_forget()
                    self.remove_button=None
            except IOError:
                messagebox.showerror("Ошибка", "Не удалось открыть изображение.")

    def show_image(self, image,label):
        # self.segment_label.config(image='')
        # self.create_interface()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертируем из BGR в RGB
        image = Image.fromarray(image)
        image.thumbnail((500, 500))  # Изменение размера для отображения
        converted_image = ImageTk.PhotoImage(image)
        # Обновляем изображение
        label.config(image=converted_image)
        label.image = converted_image 

    def canny_edge_detection(self):
        if self.original_image is not None:
            edges = cv2.Canny(self.original_image, 100, 200)
            self.show_image(edges,self.image_label)
            self.current_image = self.original_image.copy() 

    def roberts_edge_detection(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy() 
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            roberts_kernel_x = np.array([[1, 0], [0, -1]])
            roberts_kernel_y = np.array([[0, 1], [-1, 0]])
            edges_x = cv2.filter2D(gray, -1, roberts_kernel_x)
            edges_y = cv2.filter2D(gray, -1, roberts_kernel_y)
            edges = np.hypot(edges_x, edges_y)
            edges = (edges / edges.max() * 255).astype(np.uint8)
            self.show_image(edges,self.image_label)
    
    def segment_image(self):
        if self.original_image is not None:
            cluster_input = self.cluster_entry.get()
            self.current_image = self.original_image.copy() 
            
        # Проверяем, пустое ли поле ввода
            if not cluster_input:
                messagebox.showerror("Ошибка", "Вы не ввели значение. Введите значение от 2 до 20.")
                return
            try:
                n_clusters = int(cluster_input)  
                
                # Проверяем, находится ли число в допустимом диапазоне
                if n_clusters < 2 or n_clusters > 20:
                    raise ValueError("Количество кластеров должно быть от 2 до 20.")
            except ValueError as e:
                messagebox.showerror("Ошибка", f"Некорректный ввод: введите целое число от 2 до 20")
                return
            # Применяем K-средние для сегментации
            if self.remove_button is None:
                self.remove_button = tk.Button(self.button_frame, text="Убрать сегментированное изображение", command=self.remove_image)
                self.remove_button.grid(row=8,  padx=10, sticky=tk.N + tk.EW)
            Z = self.original_image.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
            _, labels, centers = cv2.kmeans(Z, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(self.original_image.shape)
            # Отображаем сегментированное изображение в новом label
            self.show_image(segmented_image, self.segment_label)
        

    def detect_keypoints(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            # Создаем объект FAST детектора
            fast = cv2.FastFeatureDetector_create()

            # Находим ключевые точки
            keypoints = fast.detect(gray_image, None)

            # Рисуем ключевые точки на изображении
            output_image = cv2.drawKeypoints(self.original_image, keypoints, None, color=(0, 255, 0))

            self.show_image(output_image , self.image_label)
    def preprocess_image(self):
    
    # Преобразуем изображение в градации серого
        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        return gray_image
    def detect_license_plate(self):
        if self.original_image is None:
            return
        preprocessed_image = self.preprocess_image()
        # Загрузка каскада Хаара для детекции номерных знаков
        plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

        # Детекция номерного знака
        plates = plate_cascade.detectMultiScale(preprocessed_image, scaleFactor=1.1, minNeighbors=5)

        # Рисуем прямоугольники вокруг найденных номерных знаков
        for (x, y, w, h) in plates:
            # Определяем координаты рамки
            plate_bbox = (x, y, w, h)
            plate_number = f"{self.current_image[y:y+h, x:x+w]}"
            # recognized_plates.append((plate_number, plate_bbox))
            cv2.rectangle(self.current_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Показываем изображение с выделенным номером
        self.show_image(self.current_image, self.image_label)
   
   


root = tk.Tk()
app = ImageApp3(root)
root.mainloop()