import albumentations as A
from pathlib import Path
from PIL import Image
import string
import cv2
import numpy as np
import glob
import os
import shutil

#aug = BEX_aug_pixel(path)
#aug.run_program()

class BEX_aug_pixel():
    
    def __init__(self, f_name: str):
        self.file_name = f_name

    def run_program(self):
        #Välj Parent folder = mappen innan yes och no
        #ex D:\Projektarbete\blandat\aug\2_split_yes\train\ == D:\Projektarbete\blandat\aug\
        fname = self.file_name
        
        data_path = Path(fname)


        #funktion som returnerar alla träningsbilder (yes/no) och dess sökväg
        path_no, path_yes = self.files_from_filenames(data_path)

        #använder funktionen get_names för att få filnamnen på alla filer
        list_of_names_no = self.get_names(path_no)
        list_of_names_yes = self.get_names(path_yes)

        #skickar in sökväg till bilder och returnerar en numpy array med bilderna i
        no_images = self.from_path_to_imglist(path_no)
        yes_images = self.from_path_to_imglist(path_yes)

        #augumenterar alla bilder och får ut dem som dict
        dict_no_aug_files = self.aug_image_and_save_to_dict(no_images, list_of_names_no)
        dict_yes_aug_files = self.aug_image_and_save_to_dict(yes_images, list_of_names_yes)

        #skapar sökväg där de aug. bilderna skall sparas
        no_aug_path = Path(str(path_no[0].parent.parent.parent)+"\\3_aug_no_train_pix\\")
        yes_aug_path = Path(str(path_yes[0].parent.parent.parent)+"\\3_aug_yes_train_pix\\")

        listan = [no_aug_path, yes_aug_path]                    #lista med yes och no sökväg
        self.check_and_create_folder(listan)                         #ta bort mapp om finns och skapa ny

        #spara de augumenterade bilderna
        saving_yes = self.saving_images(dict_yes_aug_files, yes_aug_path)
        saving_no = self.saving_images(dict_no_aug_files, no_aug_path)
        return dict_no_aug_files, dict_yes_aug_files


    def checker(self, list, word):
        """funktion som söker efter ett visst ord i en lista, returnerar sedan svaret"""
        train_paths = []
        folderpath = [n for n, x in enumerate(list) if word in x]
        
        for i in range(len(folderpath)):
            train_paths.append(list[folderpath[i]])
        return train_paths

    def files_from_filenames(self, data_path):
        """Input = platsen där de croppade bilderna ligger,
            1. hitta alla mappar med namnet train
            2. i train mappen hitta mapp som heter yes och no
            3. spara en lista på alla filnamn och dess sökväg
            4. returnerar filnamn(sökväg) per class"""
        list_of_folders = []                                        #skapar en lista av alla mappar i data_path
    
        rootdir = data_path
        for path in glob.glob(f'{rootdir}/*/**/', recursive=True):
            list_of_folders.append(path)
        
        #letar upp alla mappar som heter "train" och sparar dessa sökvägar i en lista
        training_folder_paths = self.checker(list_of_folders, 'train')   

        #letar upp vart ordet "no" finns i listan och returnerar sökvägen dit
        no_training_path = self.checker(training_folder_paths,"no")
        no_training_path = Path(no_training_path[0])

        #letar upp vart ordet "yes" finns i listan och returnerar sökvägen dit
        yes_training_path = self.checker(training_folder_paths,"yes")
        yes_training_path = Path(yes_training_path[0])
        
        #sparar ner alla filnamn som finns på destinationen
        path_no = list(no_training_path.rglob('*'))
        path_yes = list(yes_training_path.rglob('*'))
        
        return path_no, path_yes

    def get_names(self, paths):
        """funktion som sparar ner enbart filnamnet per bild"""
        return_list = []
        for i in range(len(paths)):
            return_list.append(Path(paths[i]).stem)
        return return_list

    def from_path_to_imglist(self, paths):
        """sparar ner alla sökvägar till bilder som sparas i en numpy array"""
        IMG_SIZE = (224,224)
        list_of_images = []
        for path in paths:
            img = cv2.imread(str(path))
            img = cv2.resize(img, IMG_SIZE)
            list_of_images.append(img)
        list_of_images = np.asarray(list_of_images)
        return list_of_images

    def aug_image_and_save_to_dict(self, images, list_of_names):
        """funktion som tar in alla bilderna i numpy samt alla filnamn.
        alla bilder augumenteras och dessa sparas i en lista inuti en dict per filnamn(key)"""
        dictionary = {}
        list_of_blur = [3, 5, 7, 9]
        list_of_angels = [0, 90, 180, 270]                                  
        
        for k in range(len(list_of_names)):
            #roterar bilden 4 gånger 
            for i in list_of_angels:
                transform = A.Compose([A.Affine(scale=1, 
                                        translate_percent=None, 
                                        translate_px=None, 
                                        rotate=i, 
                                        shear=None, 
                                        interpolation=1, 
                                        mask_interpolation=0, 
                                        cval=0, 
                                        cval_mask=0, 
                                        mode=0, 
                                        fit_output=False, 
                                        always_apply=False, 
                                        p=1)])
                augmented_image = transform(image=images[k])['image']            #exekverar funktionen verticalFlip
                augmented_image = Image.fromarray(augmented_image)          #konverterar bild till array
                try:                                                        #skapar en key utifrån filnamn och 
                    dictionary[list_of_names[k]] += [augmented_image]       #sparar sedan filerna i en lista
                except:
                    dictionary[list_of_names[k]] = [augmented_image]
            
            for i in range(4):
                transform = A.Compose([A.GaussianBlur(
                                        blur_limit=7, 
                                        sigma_limit=0, 
                                        always_apply=False, 
                                        p=1)])
                flip = np.asarray(dictionary[list_of_names[k]][i])          #gör bilden till en np array (vi gjorde dem till image innan)
                augmented_image = transform(image=flip)['image']            #exekverar funktionen verticalFlip
                augmented_image = Image.fromarray(augmented_image)          #konverterar bild till array
                dictionary[list_of_names[k]] += [augmented_image]           #sparar i dict            
            
            #tar de fyra roterade bilderna och flippar dem
            for i in range(4):
                transform = A.Compose([A.CLAHE(clip_limit=8.0, 
                                    tile_grid_size=(8, 8), 
                                    always_apply=False, 
                                    p=1)])
                flip = np.asarray(dictionary[list_of_names[k]][i])          #gör bilden till en np array (vi gjorde dem till image innan)
                augmented_image = transform(image=flip)['image']            #exekverar funktionen verticalFlip
                augmented_image = Image.fromarray(augmented_image)          #konverterar bild till array
                dictionary[list_of_names[k]] += [augmented_image]           #sparar i dict

            #tar de roterade och flippade bilderna och lägger på 1.3 gånger zoom
            for i in range(4):              
                transform = A.Compose([A.RandomSnow(snow_point_lower=0.5, 
                                                    snow_point_upper=0.7, 
                                                    brightness_coeff=1.3, 
                                                    always_apply=False, 
                                                    p=1)])
                zoom = np.asarray(dictionary[list_of_names[k]][i])         #gör bilden till en np array (vi gjorde dem till image innan)
                augmented_image = transform(image=zoom)['image']           #exekverar funktionen CLAHE
                augmented_image = Image.fromarray(augmented_image)          #konverterar bild till array
                dictionary[list_of_names[k]] += [augmented_image]           #sparar i dict

            for i in range(4):              
                transform = A.Compose([A.InvertImg(p=1)])
                zoom = np.asarray(dictionary[list_of_names[k]][i])         #gör bilden till en np array (vi gjorde dem till image innan)
                augmented_image = transform(image=zoom)['image']           #exekverar funktionen CLAHE
                augmented_image = Image.fromarray(augmented_image)          #konverterar bild till array
                dictionary[list_of_names[k]] += [augmented_image]           #sparar i dict

        return dictionary

    def check_and_create_folder(self, listan):
        """funktion som tittar om aug-mapp redan finns och tar isf bort den och lägger till en ny"""
        for i in listan:
            if os.path.exists(i):
                shutil.rmtree(i)
                os.mkdir(i)
            elif not os.path.exists(i):
                os.mkdir(i)
                
    def saving_images(self, dictionary, paths):
        """funktion som sparar de augumenterade bilderna i rätt mappar"""
        """varje nyckel i dict är namnet på orginalfilen och sedan är value en lista på alla de augumenterade bilderna"""
        version = string.ascii_lowercase

        for key in dictionary:                                          #key = varje nyckel i dict (filnamn)
            for j in range(len(dictionary[key])):                       #j = varje bild
                filename = f"{paths}\\{key}_{str(version[j])}.jpg"
                dictionary[key][j].save(filename)


