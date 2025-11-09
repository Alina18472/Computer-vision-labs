import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageApp2:
    def __init__(self, arg1):
        self.arg1 = arg1
        
        open_image_button = tk.Button(text="Открыть файл", command=self.open_image)
        open_image_button.grid(row=0, column=0, sticky=tk.W, padx=10)
        self.kernel = np.ones((3, 3), np.uint8)
        self.image = None
        self.new=None
        self.current_image = None 
        self.original_image = None  # Для хранения оригинального изображения
        
        # Фрейм для изображения
        self.image_frame = tk.Frame(self.arg1)
        self.image_frame.grid(row=1, column=3, sticky=tk.W+tk.N)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, padx=10, sticky=tk.N + tk.EW)

        # Фрейм для матрицы
        self.matrix_frame = tk.Frame(self.arg1)
        self.matrix_frame.grid(row=1, column=4, sticky=tk.W+tk.N,padx=10)

        
        self.size_entry = None
        self.entry_matrix = []
        # self.create_structuring_element_interface()
        self.button_frame = tk.Frame()
        self.button_frame.grid(row=1,column=0,sticky=tk.N,pady=10)

        self.size_entry = tk.Entry(self.button_frame)
        
    def restore_image(self):
        
        if self.new is not None:
            self.new = self.original_image.copy()
        if self.current_image is not None:
            self.current_image = self.original_image.copy()
        if self.new is not None:
            self.new = self.original_image.copy()
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
        # self.kernel = np.ones((3, 3), np.uint8)
        self.show_image(self.current_image)
        self.sharpness_slider.set(0)
        self.motion_blur_slider.set(1)
        self.emboss_slider.set(0)
        self.median_filter_slider.set(1)
        if self.new is not None:
            self.new = self.original_image.copy()
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
        
    def create_interface(self):
        # self.button_frame = tk.Frame()
        # self.button_frame.grid(row=1,column=0,sticky=tk.N,pady=10)
        operations = [
            ("Эрозия", self.erode_image),
            ("Дилатация", self.dilate_image),
            ("Открытие", self.open_image_morph),
            ("Закрытие", self.close_image),
            ("Градиент", self.gradient_image),
            ("Цилиндр", self.cylinder_image),
            ("Чёрная шляпа", self.black_hat_image),
            
            
        ]
        for i, (text, command) in enumerate(operations):
            button = tk.Button(self.button_frame,text=text, command=command)
            button.grid(row=i+1,  pady=3,padx=10,sticky=tk.W)
        # self.structure_frame = tk.Frame()
        # self.structure_frame.grid(row=3,column=0)
        tk.Button(self.button_frame,text="Сбросить", command=self.restore_image).grid(row=8, padx=10,sticky=tk.W)
        tk.Label(self.button_frame, text="Введите размер структурного элемента:").grid(row=9, padx=10,sticky=tk.W)
        
        self.size_entry.grid(row=10, padx=10,sticky=tk.W)
        tk.Button(self.button_frame, text="Создать матрицу", command=self.create_matrix_entries).grid(row=11, padx=10, columnspan=2, pady=10,sticky=tk.W)
        
        
        self.slider_frame = tk.Frame()
        self.slider_frame.grid(row=1, column=1,sticky=tk.N)
        
        self.sharpness_slider = tk.Scale(self.slider_frame, from_=0, to=10, orient=tk.HORIZONTAL,label="Резкость", command=self.update_image)
        self.sharpness_slider.grid(row=1,sticky=tk.W)
        self.sharpness_slider.set(0) 
        # Ползунок для размытия в движении
        self.motion_blur_slider = tk.Scale(self.slider_frame, from_=1, to=100, orient=tk.HORIZONTAL, label="Размытие", command=self.update_image)
        self.motion_blur_slider.grid(row=2,sticky=tk.W)

        # Ползунок для тиснения
        self.emboss_slider = tk.Scale(self.slider_frame, from_=0, to=10, orient=tk.HORIZONTAL, label="Тиснение", command=self.update_image)
        self.emboss_slider.grid(row=3,sticky=tk.W)

        # # Ползунок для медианной фильтрации
        self.median_filter_slider = tk.Scale(self.slider_frame, from_=1, to=30, orient=tk.HORIZONTAL, label="Фильтрация", command=self.update_image)
        self.median_filter_slider.grid(row=4,sticky=tk.W)    
    

    def create_matrix_entries(self):
        try:
            size = int(self.size_entry.get())
            if size <=0:
                messagebox.showerror("Ошибка", "Размер должен быть положительным.")
                return
            if size %2!=1:
                messagebox.showerror("Ошибка", "Размер должен быть нечетным.")
                return
            
            # Удаление предыдущих полей ввода
            for widget in self.matrix_frame.winfo_children():
                widget.destroy()  # Очистка предыдущих записей
            
            self.entry_matrix.clear()  # Очистка предыдущих полей ввода
            # tk.Button(self.matrix_frame,text="Заполнить единицами", command=self.fill).grid(row=0, padx=10,sticky=tk.W)
            for i in range(size):
                row_entries = []
                for j in range(size):
                    entry = tk.Entry(self.matrix_frame, width=5)
                    entry.insert(0, '0')  # Заполнение нулями
                    entry.grid(row=i, column=j,sticky=tk.N)
                    row_entries.append(entry)
                self.entry_matrix.append(row_entries)
            
            tk.Button(self.button_frame, text="Создать структурный элемент", command=self.create_structuring_element).grid(row=12, padx=10, columnspan=2,sticky=tk.W)
            tk.Button(self.button_frame, text="Заполнить единицами", command=self.fill_with_ones).grid(row=13, column=0, padx=10,pady=10, sticky=tk.W)
            tk.Button(self.button_frame, text="Заполнить нулями", command=self.fill_with_zeros).grid(row=14, column=0, padx=10, sticky=tk.W)
            tk.Button(self.button_frame, text="По умолчанию", command=self.default).grid(row=15, column=0, padx=10,pady=10, sticky=tk.W)
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное число.")
  
    def default(self):
        self.kernel = np.ones((3, 3), np.uint8)
        # self.size_entry.delete(0, tk.END)
        self.clear_matrix_and_buttons()
    def fill_with_ones(self):
        for row in self.entry_matrix:
            for entry in row:
                entry.delete(0, tk.END)  # Очистка текущего значения
                entry.insert(0, '1')  # Заполнение единицей

    def fill_with_zeros(self):
        for row in self.entry_matrix:
            for entry in row:
                entry.delete(0, tk.END)  # Очистка текущего значения
                entry.insert(0, '0')  # Заполнение нулем
    def create_structuring_element(self):
        try:
            
            size = int(self.size_entry.get())
            if size>0:
                self.kernel = np.zeros((size, size), dtype=np.uint8)  # Сохраняем структурный элемент в атрибут
                for i in range(size):
                    for j in range(size):
                        value = int(self.entry_matrix[i][j].get())
                        self.kernel[i][j] = value
                
                messagebox.showinfo("Информация", "Структурный элемент создан.")
            else:
                messagebox.showerror("Ошибка", "Размер должен быть положительным.")
                
        except Exception as e:
            messagebox.showerror("Ошибка", "Введено некорректное значение")
        
    def clear_matrix_and_buttons(self):
        # Удаление предыдущих полей ввода
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()  # Очистка предыдущих записей
        
        self.entry_matrix.clear()  # Очистка предыдущих полей ввода
        self.size_entry.delete(0, tk.END)
        # Удаление кнопок
        widgets = self.button_frame.winfo_children()
    
    
        for i in range(11, len(widgets)):
            widgets[i].destroy()  # Удаление виджета

    def open_image(self):
        self.clear_matrix_and_buttons()
        
        file_path = filedialog.askopenfilename(
            filetypes=[("JPEG", "*.jpg;*.jpeg"), ("PNG", "*.png"), ("GIF", "*.gif"), ("Все файлы", "*.*"),])
        if file_path:
            try:
                self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Чтение в градациях серого
                self.current_image = self.original_image.copy() 
                self.new=None
                self.show_image(self.original_image)
                self.kernel = np.ones((3, 3), np.uint8)
                self.create_interface()
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def show_image(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Конвертация в RGB для отображения
        image = Image.fromarray(image)
        image.thumbnail((500, 500))  # Изменение размера для отображения
        converted_image = ImageTk.PhotoImage(image)

        # Обновляем изображение
        self.image_label.config(image=converted_image)
        self.image_label.image = converted_image  # Сохраняем ссылку на изображение

    def update_current_image(self, new_image):
        if self.current_image is None:
            self.current_image = new_image
        else:
            self.current_image = new_image  # Обновляем текущее изображение
        self.new = new_image

    def update_image(self, new_image):
        if self.original_image is not None:
            # Получаем значения из ползунков
            motion_blur_value = self.motion_blur_slider.get()
            emboss_value = self.emboss_slider.get()
            median_filter_value = self.median_filter_slider.get()
            sharpness_value = self.sharpness_slider.get()

            # Начинаем с оригинального изображения
            if self.new is None:
                self.current_image = self.original_image.copy()
            else:
                self.current_image = self.new.copy()

            # Применяем эффекты по порядку
            if motion_blur_value > 0:
                kernel_size = (motion_blur_value // 2) * 2 + 1
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                kernel /= kernel_size
                blur_image = cv2.filter2D(self.current_image, -1, kernel)

            if emboss_value > 0:
                kernel = np.array([[-2* emboss_value, -1 * emboss_value, 0],
                                   [-1 * emboss_value, 1, 1* emboss_value],
                                   [0, 1 * emboss_value, 2* emboss_value]])
                emboss_image = cv2.filter2D(blur_image, -1, kernel)
            else:
                emboss_image = False

            if median_filter_value > 0 and emboss_image is not False:
                if median_filter_value % 2 == 0:
                    median_filter_value += 1
                median_image = cv2.medianBlur(emboss_image, median_filter_value)
            else:
                if median_filter_value % 2 == 0:
                    median_filter_value += 1
                median_image = cv2.medianBlur(blur_image, median_filter_value)
            # Применяем эффект резкости
            if sharpness_value > 0:
                value = int(sharpness_value)
                kernel = np.array([[0, -1 * value, 0],
                                   [-1 * value, 4 * value + 1, -1 * value],
                                   [0, -1 * value, 0]])
                final_image = cv2.filter2D(median_image, -1, kernel)
                self.current_image = final_image
                self.show_image(final_image)
            else:
                
                self.current_image = median_image
                # Обновляем текущее изображение
                self.show_image(median_image)

    def erode_image(self):
        if self.original_image is not None:
            if self.kernel is not None:
                k=self.kernel
            else:
                k = np.ones((3, 3), np.uint8)  # Структурный элемент
            eroded = cv2.erode(self.current_image, k)
            self.update_current_image(eroded)
            self.show_image(self.current_image)

    def dilate_image(self):
        if self.original_image is not None:
            if self.kernel is not None:
                k=self.kernel
            else:
                k = np.ones((3, 3), np.uint8)  # Структурный элемент
            dilated = cv2.dilate(self.current_image, k)
            self.update_current_image(dilated)
            self.show_image(self.current_image)

    def open_image_morph(self):
        if self.original_image is not None:
            if self.kernel is not None:
                k=self.kernel
            else:
                k = np.ones((3, 3), np.uint8)  # Структурный элемент
            opened = cv2.morphologyEx(self.current_image, cv2.MORPH_OPEN, k)
            self.update_current_image(opened)
            self.show_image(self.current_image)

    def close_image(self):
        if self.original_image is not None:
            if self.kernel is not None:
                k=self.kernel
            else:
                k = np.ones((3, 3), np.uint8)  # Структурный элемент
            closed = cv2.morphologyEx(self.current_image, cv2.MORPH_CLOSE, k)
            self.update_current_image(closed)
            self.show_image(self.current_image)

    def gradient_image(self):
        if self.original_image is not None:
            if self.kernel is not None:
                k=self.kernel
            else:
                k= np.ones((3, 3), np.uint8)  # Структурный элемент
            gradient = cv2.morphologyEx(self.current_image, cv2.MORPH_GRADIENT, k)
            self.update_current_image(gradient)
            self.show_image(self.current_image)

    def cylinder_image(self):
        if self.original_image is not None:
            if self.kernel is not None:
                k = self.kernel
            else:
                k = np.ones((3, 3), np.uint8)  # Структурный элемент
            opened = cv2.morphologyEx(self.current_image, cv2.MORPH_OPEN, k)
            cylinder = cv2.subtract(self.current_image, opened)
            self.update_current_image(cylinder)
            self.show_image(self.current_image)

    def black_hat_image(self):
         if self.original_image is not None:
            if self.kernel is not None:
                k = self.kernel
            else:
                k = np.ones((2, 2), np.uint8)  # Структурный элемент
            closed = cv2.morphologyEx(self.current_image, cv2.MORPH_CLOSE, k)
            black_hat = cv2.subtract(closed, self.current_image)
            self.update_current_image(black_hat)
            self.show_image(self.current_image)

     

    
    

   


root = tk.Tk()
app = ImageApp2(root)
root.mainloop()