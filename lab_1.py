import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import os
import exifread
import matplotlib.pyplot as matp
import numpy as np
import io

class ImageApp:
    def __init__(self, arg1):
        self.arg1 = arg1
        arg1.title("Приложение для взаимодействия с изображением.")
        
        open_image_button = tk.Button(text="Открыть файл", command=self.open_image)
        open_image_button.grid(row=0, column=0, sticky=tk.W,padx=10)

        self.image = None
        self.original_image = None  # Для хранения оригинального изображения
        self.original_grey_image=None
        self.restored_image =None
        self.image_frame = tk.Frame(arg1)
        self.image_frame.grid(row=1, column=1, sticky=tk.W)
        #место для изображения     
        self.image_label = tk.Label(self.image_frame)
        self.image_label.grid(row=1, column=1, padx=10, pady=10,sticky=tk.N + tk.EW)
        #информация об изображениии
        self.info_text = tk.Text(self.image_frame,height=18, width=50)
        self.button_frame = tk.Frame(arg1)
        self.button_frame.grid(row=1, column=0, sticky=tk.W)
        #преобразование в серый
        self.restore_button= tk.Button(self.button_frame,text="Восстановить изображение", command=self.restore)
        
       
        self.gray_button = tk.Button(self.button_frame,text="Преобразовать в черно-белый", command=self.convert_to_grayscale)
        self.scale_frame = tk.Frame(arg1)
        self.scale_frame.grid(row=2, column=0,sticky = tk.W)
        # Ползунки яркости, нысыщенности и контрастности
        self.brightness_scale = tk.Scale(self.button_frame, from_=0, to=2, resolution=0.1, label="Яркость", orient=tk.HORIZONTAL, command=self.update_image)
        self.brightness_scale.set(1)  
       
        self.saturation_scale = tk.Scale(self.button_frame, from_=0, to=2, resolution=0.1, label="Насыщенность", orient=tk.HORIZONTAL, command=self.update_image)
        self.saturation_scale.set(1)  
        
        self.contrast_scale = tk.Scale(self.button_frame, from_=0, to=2, resolution=0.1, label="Контрастность", orient=tk.HORIZONTAL, command=self.update_image)
        self.contrast_scale.set(1)  
        
        # Кнопки для отображения гистограммы
   
        self.histogram_button_before = tk.Button(self.button_frame,text="Гистограмма до изменений", command=self.show_original_histogram)
        self.histogram_button_after = tk.Button(self.button_frame,text="Гистограмма после изменений", command=self.show_histogram_after)

        # Создаем Label для гистограмм
        self.histogram_frame = tk.Frame(arg1)
        self.histogram_frame.grid(row=1, column=2, sticky=tk.W)
        self.histogram_label_before = tk.Label(self.histogram_frame)
        self.histogram_label_before.grid(row=1, column=2)  # Позиция для гистограммы до

        self.histogram_label_after = tk.Label(self.histogram_frame)
        self.histogram_label_after.grid(row=2, column=2) 
        
        self.red_hist_button = tk.Button(self.button_frame,text="Гистограмма красного канала", command=self.show_red_histogram)
        self.green_hist_button = tk.Button(self.button_frame,text="Гистограмма зеленого канала", command=self.show_green_histogram)
        self.blue_hist_button = tk.Button(self.button_frame,text="Гистограмма синего канала", command=self.show_blue_histogram)
        self.histogram_button = tk.Button(self.button_frame, text="Показать гистограммы в окне", command=self.show_all_histograms)
        
        self.linear_button = tk.Button(self.button_frame,text="Применить линейную коррекцию", command=self.apply_linear_correction)
        
        self.non_linear_button = tk.Button(self.button_frame,text="Применить нелинейную коррекцию", command=self.apply_non_linear_correction)
        self.info_button = tk.Button(self.button_frame,text="Скрыть информацию", command=self.hide_info)
        
        self.is_grayscale = False
        self.is_hid=False
    def hide_info(self):
        if self.info_text:
            if not self.is_hid:  
                self.info_text.grid_forget()
                self.info_button.config(text="Отобразить информацию")  # Меняем текст кнопки
   
            else:  
                self.info_text.grid(row=2,column=1) # Возвращаем оригинал
                self.info_button.config(text="Скрыть информацию")  # Меняем текст кнопки
                
                
            self.is_hid = not self.is_hid  # Меняем состояние
            
    def show_all_histograms(self):
        if self.image:
            img_array = np.array(self.image)

            # Создаем фигуру с четырьмя подграфиками
            fig, axs = matp.subplots(4, 1, figsize=(5, 10))

            # Гистограмма красного канала
            r_channel = img_array[:, :, 0].flatten()
            axs[0].hist(r_channel, bins=256, color='red', alpha=0.6)
            axs[0].set_xlim([0, 256])
            axs[0].set_title('Гистограмма Красного Канала')
            axs[0].set_xlabel('Интенсивность')
            axs[0].set_ylabel('Количество пикселей')

            # Гистограмма зеленого канала
            g_channel = img_array[:, :, 1].flatten()
            axs[1].hist(g_channel, bins=256, color='green', alpha=0.6)
            axs[1].set_xlim([0, 256])
            axs[1].set_title('Гистограмма Зеленого Канала')
            axs[1].set_xlabel('Интенсивность')
            axs[1].set_ylabel('Количество пикселей')

            # Гистограмма синего канала
            b_channel = img_array[:, :, 2].flatten()
            axs[2].hist(b_channel, bins=256, color='blue', alpha=0.6)
            axs[2].set_xlim([0, 256])
            axs[2].set_title('Гистограмма Синего Канала')
            axs[2].set_xlabel('Интенсивность')
            axs[2].set_ylabel('Количество пикселей')

            # Гистограмма всех трех каналов
            axs[3].hist(r_channel, bins=256, color='red', alpha=0.5, label='Красный')
            axs[3].hist(g_channel, bins=256, color='green', alpha=0.5, label='Зеленый')
            axs[3].hist(b_channel, bins=256, color='blue', alpha=0.5, label='Синий')
            axs[3].set_xlim([0, 256])
            axs[3].set_title('Гистограмма RGB')
            axs[3].set_xlabel('Интенсивность')
            axs[3].set_ylabel('Количество пикселей')
            axs[3].legend()

            matp.tight_layout()
            matp.show()

    def restore(self):
        restored_img= self.restored_image
        self.image=restored_img
        self.original_image=restored_img
        self.is_grayscale = False
        self.gray_button.config(text="Преобразовать в черно-белый")
        self.saturation_scale.set(1)
        self.brightness_scale.set(1)
        self.contrast_scale.set(1)
        self.linear_button.grid_forget()
        self.non_linear_button.grid_forget()
        self.saturation_scale.grid(row=16, column=0, sticky=tk.W,padx=10)
        self.saturation_scale.set(1)
        self.red_hist_button.grid(row=8, column=0, sticky=tk.W,padx=10)
        self.green_hist_button.grid(row=9, column=0, sticky=tk.W,padx=10)
        self.blue_hist_button.grid(row=10, column=0, sticky=tk.W,padx=10)
        self.histogram_button.grid(row=11, column=0, sticky=tk.W,padx=10)
        self.info_button.grid(row=12, column=0, sticky=tk.W,padx=10)
        if hasattr(self, 'histogram_label'):
            self.histogram_label.image = None
        if hasattr(self, 'histogram_label_before'):
            self.histogram_label_before.image = None
        if hasattr(self, 'histogram_label_after'):
            self.histogram_label_after.image = None
        self.show_image()
    def apply_linear_correction(self):
        if self.image:
            img_array = np.array(self.image)
            # Получаем минимальные и максимальные значения пикселей
            x_min = img_array.min()
            x_max = img_array.max()
            # Применяем линейную коррекцию ко всем пикселям
            corrected_array = (img_array - x_min) * (255 // (x_max - x_min))
            corrected_array = np.clip(corrected_array, 0, 255)  # Ограничиваем значения до [0, 255]
            corrected_array = corrected_array.astype(np.uint8)  # Приводим к типу uint8
            # Создаем новое изображение из скорректированного массива
            self.image = Image.fromarray(corrected_array)
            self.show_image()


    def apply_non_linear_correction(self):
        # Преобразуем изображение в массив NumPy
        
        brightness = self.brightness_scale.get()
        contrast = self.contrast_scale.get()
        gamma=1
        if brightness > 1:
            gamma=2
        if brightness <1:
            gamma=1/2 
        # if contrast > 1 and brightness==1:
        #     gamma =1/2
        # if contrast < 1 and brightness==1:
        #     gamma=2
        img_array = np.array(self.image).astype(np.float32) / 255.0
        # Применяем гамма-коррекцию
        img_array = np.clip(np.power(img_array, gamma), 0, 1) * 255
        # Приводим к типу uint8
        corrected_image = Image.fromarray(img_array.astype(np.uint8))
        self.image = corrected_image
        self.show_image()

    def convert_to_grayscale(self):
        if self.image:
            if not self.is_grayscale:  # Если изображение не в черно-белом режиме
                self.original_image = self.image.copy()  # Сохраняем оригинальное изображение
                self.image = ImageOps.grayscale(self.image)  # Преобразуем в черно-белый
                self.gray_button.config(text="Вернуть прежний цвет")  # Меняем текст кнопки
                self.linear_button.grid(row=11, column=0,sticky=tk.W,padx=10)
                self.non_linear_button.grid(row=12, column=0,sticky=tk.W,padx=10)
                self.info_button.grid(row=13, column=0, sticky=tk.W,padx=10)
                self.saturation_scale.grid_forget()
                self.red_hist_button.grid_forget()
                self.green_hist_button.grid_forget()
                self.blue_hist_button.grid_forget()
                self.histogram_button.grid_forget()
                if hasattr(self, 'histogram_label'):
                    self.histogram_label.image = None
                if hasattr(self, 'histogram_label_before'):
                    self.histogram_label_before.image = None
                if hasattr(self, 'histogram_label_after'):
                    self.histogram_label_after.image = None
                
                
            else:  # Если изображение уже в черно-белом режиме
                self.image = self.original_image  # Возвращаем оригинал
                self.gray_button.config(text="Преобразовать в черно-белый")  # Меняем текст кнопки
                self.linear_button.grid_forget()
                self.non_linear_button.grid_forget()
                self.saturation_scale.grid(row=16, column=0, sticky=tk.W,padx=10)
                self.saturation_scale.set(1)
                self.red_hist_button.grid(row=8, column=0, sticky=tk.W,padx=10)
                self.green_hist_button.grid(row=9, column=0, sticky=tk.W,padx=10)
                self.blue_hist_button.grid(row=10, column=0, sticky=tk.W,padx=10)
                self.histogram_button.grid(row=11, column=0, sticky=tk.W,padx=10)
                self.info_button.grid(row=12, column=0, sticky=tk.W,padx=10)
                if hasattr(self, 'histogram_label'):
                    self.histogram_label.image = None
                if hasattr(self, 'histogram_label_before'):
                    self.histogram_label_before.image = None
                if hasattr(self, 'histogram_label_after'):
                    self.histogram_label_after.image = None
                
            self.is_grayscale = not self.is_grayscale  # Меняем состояние
            self.show_image()

    def update_image(self, _):
        if self.original_image:
            # Применяем изменения к изображению
            brightness = self.brightness_scale.get()
            saturation = self.saturation_scale.get()
            contrast = self.contrast_scale.get()

            # Применяем яркость
            if self.is_grayscale ==False:
                enhancer1 = ImageEnhance.Brightness(self.original_image) #принимает оригинальное изображение (self.original_image) и создает объект-усилитель.
            else:
                enhancer1 = ImageEnhance.Brightness(self.original_grey_image)
            img_brightened = enhancer1.enhance(brightness)

            # Применяем насыщенность
            enhancer2 = ImageEnhance.Color(img_brightened)
            img_saturated = enhancer2.enhance(saturation)

            # Применяем контрастность
            enhancer3 = ImageEnhance.Contrast(img_saturated)
            final_image = enhancer3.enhance(contrast)
            self.image=final_image
            # Обновляем изображение на экране
            self.show_image()

    def show_red_histogram(self):
        
        if self.image:
            # Преобразуем изображение в массив NumPy
            img_array = np.array(self.image)

            fig = matp.figure(figsize=(5, 2))
            
            if self.is_grayscale:
                # Если изображение черно-белое, создаем гистограмму на основе одного канала
                gray_channel = img_array.flatten()  
                matp.hist(gray_channel, bins=256, color='gray', alpha=0.6)
                matp.title('Гистограмма Черно-Белого Изображения')
            else:
                # Если изображение цветное, создаем гистограмму для красного канала
                r_channel = img_array[:, :, 0].flatten()  # Красный канал
                matp.hist(r_channel, bins=256, color='red', alpha=0.6)
                matp.title('Гистограмма Красного Канала')

            matp.xlim([0, 256])
            matp.xlabel('Интенсивность')
            matp.ylabel('Количество пикселей')
            matp.tight_layout()
            
            # Сохраняем гистограмму в буфер
            buf = io.BytesIO()
            matp.savefig(buf, format='png')
            buf.seek(0)  # Перемещаем курсор в начало буфера
            histogram_image = Image.open(buf)
            histogram_image.thumbnail((500, 200))  # Изменяем размер по необходимости
            histogram_tk = ImageTk.PhotoImage(histogram_image)

            if hasattr(self, 'histogram_label'):
                self.histogram_label.config(image=histogram_tk)
                self.histogram_label.image = histogram_tk  
            else:
                
                self.histogram_label = tk.Label(self.histogram_frame,image=histogram_tk)
                self.histogram_label.image = histogram_tk  
                self.histogram_label.grid(row=3, column=2) 
            
            matp.close(fig)
    
    def show_green_histogram(self):
        if self.image:
            img_array = np.array(self.image)
            

            fig=matp.figure(figsize=(5, 2))
            if self.is_grayscale:
                # Если изображение черно-белое, создаем гистограмму на основе одного канала
                gray_channel = img_array.flatten()  # Плоский массив значений
                matp.hist(gray_channel, bins=256, color='gray', alpha=0.6)
                matp.title('Гистограмма Черно-Белого Изображения')
            else:
                g_channel = img_array[:, :, 1].flatten()
                matp.hist(g_channel, bins=256, color='green', alpha=0.6)
                matp.title('Гистограмма Зеленого Канала')
            
            matp.xlim([0, 256])
            
            matp.xlabel('Интенсивность')
            matp.ylabel('Количество пикселей')
            matp.tight_layout()
             # Сохраняем гистограмму в буфер
            buf = io.BytesIO()
            matp.savefig(buf, format='png')
            buf.seek(0) 

            # Преобразуем буфер в изображение
            histogram_image = Image.open(buf)

            # Отображаем изображение в приложении
            histogram_image.thumbnail((500, 200))  
            histogram_tk = ImageTk.PhotoImage(histogram_image)

            if hasattr(self, 'histogram_label'):
                self.histogram_label.config(image=histogram_tk)
                self.histogram_label.image = histogram_tk 
            else:
            # Если Label еще не создан, создаем его
                self.histogram_label = tk.Label(self.histogram_frame, image=histogram_tk)
                self.histogram_label.image = histogram_tk 
                self.histogram_label.grid(row = 3,column=2)  

            matp.close(fig) 
            

    def show_blue_histogram(self):
        if self.image:
            img_array = np.array(self.image)

            fig=matp.figure(figsize=(5, 2))
            if self.is_grayscale:
                # Если изображение черно-белое, создаем гистограмму на основе одного канала
                gray_channel = img_array.flatten() 
                matp.hist(gray_channel, bins=256, color='gray', alpha=0.6)
                matp.title('Гистограмма Черно-Белого Изображения')
            else:
                b_channel = img_array[:, :, 2].flatten()
                matp.hist(b_channel, bins=256, color='blue', alpha=0.6)
                matp.title('Гистограмма Синего Канала')
            
            matp.xlim([0, 256])
          
            matp.xlabel('Интенсивность')
            matp.ylabel('Количество пикселей')
            matp.tight_layout()
             # Сохраняем гистограмму в буфер
            buf = io.BytesIO()
            matp.savefig(buf, format='png')
            buf.seek(0)  # Перемещаем курсор в начало буфера

            # Преобразуем буфер в изображение
            histogram_image = Image.open(buf)

            # Отображаем изображение в приложении
            histogram_image.thumbnail((500, 200)) 
            histogram_tk = ImageTk.PhotoImage(histogram_image)

            if hasattr(self, 'histogram_label'):
                self.histogram_label.config(image=histogram_tk)
                self.histogram_label.image = histogram_tk  
            else:
            # Если Label еще не создан, создаем его
                self.histogram_label = tk.Label(self.histogram_frame, image=histogram_tk)
                self.histogram_label.image = histogram_tk 
                self.histogram_label.grid(row = 3,column=2)  

            matp.close(fig) 
    def show_original_histogram(self):
        if self.original_image:
            # Преобразуем изображение в массив NumPy
            img_array = np.array(self.original_image)

            # Создаем фигуру для гистограммы
            fig, ax = matp.subplots(figsize=(5, 2))

            # Проверяем количество каналов
            if self.is_grayscale:
              
                gray_channel = img_array.flatten()
                ax.hist(gray_channel, bins=256, color='gray', alpha=0.6, label='Gray')
                ax.set_title('Гистограмма Черно-Белого Изображения')
            else:  
                # Разделяем каналы R, G и B
                r_channel = img_array[:, :, 0].flatten()
                g_channel = img_array[:, :, 1].flatten()
                b_channel = img_array[:, :, 2].flatten()

                ax.hist(r_channel, bins=256, color='red', alpha=0.6, label='Red')
                ax.hist(g_channel, bins=256, color='green', alpha=0.6, label='Green')
                ax.hist(b_channel, bins=256, color='blue', alpha=0.6, label='Blue')
                ax.set_title('Гистограмма Цветовых Каналов')

            ax.set_xlim([0, 256])
            ax.set_xlabel('Интенсивность')
            ax.set_ylabel('Количество пикселей')
            ax.legend(loc='upper right')

            # Сохраняем гистограмму в буфер
            buf = io.BytesIO()
            matp.tight_layout()
            matp.savefig(buf, format='png')
            buf.seek(0)  

            # Преобразуем буфер в изображение
            histogram_image = Image.open(buf)

            # Отображаем изображение в приложении
            histogram_image.thumbnail((500, 200)) 
            histogram_tk = ImageTk.PhotoImage(histogram_image)

            # Обновляем соответствующий Label с гистограммой
            self.histogram_label_before.config(image=histogram_tk)
            self.histogram_label_before.image = histogram_tk  

            matp.close(fig)   
    def show_histogram_after(self):
        if self.image:
            # Преобразуем изображение в массив NumPy
            img_array = np.array(self.image)
            
        
            # Создаем фигуру для гистограммы
            fig, ax = matp.subplots(figsize=(5, 2))
            if self.is_grayscale:
            # Если изображение черно-белое, создаем гистограмму для единственного канала
                gray_channel = img_array.flatten()
                ax.hist(gray_channel, bins=256, color='gray', alpha=0.6, label='Gray')
                ax.set_title('Гистограмма Черно-Белого Изображения После')
            else:
                # Если изображение цветное, разделяем каналы R, G и B
                r_channel = img_array[:, :, 0].flatten()
                g_channel = img_array[:, :, 1].flatten()
                b_channel = img_array[:, :, 2].flatten()

                ax.hist(r_channel, bins=256, color='red', alpha=0.6, label='Red')
                ax.hist(g_channel, bins=256, color='green', alpha=0.6, label='Green')
                ax.hist(b_channel, bins=256, color='blue', alpha=0.6, label='Blue')
                ax.set_title('Гистограмма Цветовых Каналов После')
            
            
            ax.set_xlim([0, 256])
            
            ax.set_xlabel('Интенсивность')
            ax.set_ylabel('Количество пикселей')
            ax.legend(loc='upper right')

             # Сохраняем гистограмму в буфер
            buf = io.BytesIO()
            matp.tight_layout()
            matp.savefig(buf, format='png')
            buf.seek(0)  

            # Преобразуем буфер в изображение
            histogram_image = Image.open(buf)

            # Отображаем изображение в приложении
            histogram_image.thumbnail((500, 200)) 
            histogram_tk = ImageTk.PhotoImage(histogram_image)

            # Обновляем соответствующий Label с гистограммой
            self.histogram_label_after.config(image=histogram_tk)
            self.histogram_label_after.image = histogram_tk 

            matp.close(fig)
    
    

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("JPEG", "*.jpg;*.jpeg"),
                ("PNG", "*.png"),
                ("GIF", "*.gif"),
                ("Все файлы", "*.*"),
            ]
        )
        if file_path:
            try:
                self.image = Image.open(file_path)
                self.original_image = self.image.copy()  
                self.restored_image = self.image.copy() 
                self.first_state_image = self.image.copy()
                self.show_image()
                self.show_image_info(file_path)
                
                self.original_grey_image=ImageOps.grayscale(self.image)
                
               # self.gray_button.pack(side=tk.BOTTOM)
               # #показываем кнопки после загрузки
                self.info_text.grid(row=2, column=1, padx=10, pady=10)
                self.restore_button.grid(row=2,column=0,sticky=tk.W,padx=10)
                
                self.gray_button.grid(row=3, column=0, sticky=tk.W,padx=10)
                
                self.brightness_scale.grid(row=15, column=0, sticky=tk.W,padx=10)
                self.saturation_scale.grid(row=16, column=0, sticky=tk.W,padx=10)
                self.contrast_scale.grid(row=17, column=0, sticky=tk.W,padx=10)
                
                
                self.histogram_button_before.grid(row=4, column=0, sticky=tk.W,padx=10)
                self.histogram_button_after.grid(row=5, column=0, sticky=tk.W,padx=10)
                

                self.red_hist_button.grid(row=6, column=0, sticky=tk.W,padx=10)
                self.green_hist_button.grid(row=7, column=0, sticky=tk.W,padx=10)
                self.blue_hist_button.grid(row=8, column=0, sticky=tk.W,padx=10)
                self.histogram_button.grid(row=9, column=0, sticky=tk.W,padx=10)
                self.info_button.grid(row=10, column=0, sticky=tk.W,padx=10)

                
                # Сброс состояния
                self.is_grayscale = False
                self.gray_button.config(text="Преобразовать в черно-белый") 
                self.brightness_scale.set(1)
                self.saturation_scale.set(1)
                self.contrast_scale.set(1)
                self.histogram_label_before.image = None
                self.histogram_label_after.image = None 
                self.histogram_label.image = None 
            
            except IOError:
                messagebox.showerror("Ошибка", "Не удалось открыть изображение.")

    def show_image(self):
        self.image.thumbnail((500, 500))  # Изменение размера для отображения
        converted_image = ImageTk.PhotoImage(self.image)  # Преобразует изображение Pillow в формат для Tkinter
        self.image_label.config(image=converted_image)  # Обновляет виджет, чтобы отобразить новое изображение
        self.image_label.image = converted_image  # Сохраняет ссылку на объект converted_image
        
            
        

   

    def show_image_info(self, file_path):
        depth_of_color = len(self.image.getbands()) * 8
        bit_word = get_bit_word(depth_of_color)

        self.info_text.delete("1.0", tk.END)  # Очистка текстового поля.
        self.info_text.insert(tk.END, f"Размер, занимаемый на диске: {os.path.getsize(file_path)} байт\n")
        self.info_text.insert(tk.END, f"Разрешение: {self.image.size[0]}x{self.image.size[1]}\n")
        self.info_text.insert(tk.END, f"Глубина цвета: {depth_of_color} {bit_word}\n")
        self.info_text.insert(tk.END, f"Формат файла: {self.image.format}\n")
        self.info_text.insert(tk.END, f"Цветовая модель: {self.image.mode}\n")
        self.info_text.insert(tk.END,f"\nИнформация EXIF.\n")
        
        # EXIF данные
        
        with open(file_path, 'rb') as f:  # Открываем файл изображения в бинарном режиме
            tags = exifread.process_file(f)
        exif_info = {  # Словарь с ключами и значениями, полученными из EXIF данных
            'Модель камеры': str(tags.get('Image Model')) if 'Image Model' in tags else 'Не указано',
            'Производитель камеры': str(tags.get('Image Make')) if 'Image Make' in tags else 'Не указано','Дата съемки': str(tags.get('Image DateTime')) if 'Image DateTime' in tags else 'Не указано',
            'Авторское право': str(tags.get('Image Copyright')) if 'Image Copyright' in tags else 'Не указано',
            'Выдержка': str(tags.get('EXIF ExposureTime')) if 'EXIF ExposureTime' in tags else 'Не указано',
            'Диафрагма': str(tags.get('EXIF FNumber')) if 'EXIF FNumber' in tags else 'Не указано',
            'ISO': str(tags.get('EXIF ISOSpeedRatings')) if 'EXIF ISOSpeedRatings' in tags else 'Не указано',
            #'Разрешение сенсора': str(tags.get('EXIF SensorWidth')) + 'x' + str(tags.get('EXIF SensorHeight')) if 'EXIF SensorWidth' in tags and 'EXIF SensorHeight' in tags else 'Не указано',
            }
        for key, value in exif_info.items():
            self.info_text.insert(tk.END, f"{key}: {value}\n")  
        
        

        self.info_text.insert(tk.END, f"\nДополнительная информация.")
        self.info_text.insert(tk.END, f"\nИмя файла: {os.path.basename(file_path)}\n")
        
        
def get_bit_word(count):
            if count % 10 == 1 and count % 100 != 11:
                return "бит"
            elif count % 10 in [2, 3, 4] and not (count % 100 in [12, 13, 14]):
                return "бита"
            else:
                return "бит"
            
root = tk.Tk()
app = ImageApp(root)
root.mainloop()