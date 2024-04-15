import os

# Укажите путь к каталогу с изображениями
directory = r"C:\Users\Mysty\Downloads\DATA_Maguire_20180517_ALL\P\CP"

# Пройдите по всем файлам в каталоге
for filename in os.listdir(directory):
    # Проверьте, является ли файл изображением (по расширению .jpg)
    if filename.endswith(".jpg"):
        # Создайте новое имя файла, добавив "new_" к началу
        new_filename = "cracked_" + filename
        # Переименуйте файл
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))