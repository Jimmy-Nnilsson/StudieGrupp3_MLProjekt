import numpy as np
import os
import cv2
from pathlib import Path
from os.path import exists
import glob

#The program input must be to a folder where there is a yes and no folder

class BEX_check_remove_dups():

    def __init__(self, f_name: str):
        self.file_name = f_name

    def run_program(self):
        fname = self.file_name
        data_path = Path(fname)
        
        #här får vi listor på alla filer och dess sökväg per class
        path_no, path_yes = self.files_from_filenames(data_path)

        #skapar np.array av bilderna per class
        no_image_list = self.images_to_list(path_no)
        yes_image_list = self.images_to_list(path_yes)

        #första variabeln = dict med alla bilder som finns duplicerade samt dess pixelvärde (sorterad)
        #andra variabeln = lista över alla filer som vi inte hittade någon duplicering av
        get_no_duplicates, no_without_dup = self.check_duplicates(no_image_list, path_no)
        get_yes_duplicates, yes_without_dup = self.check_duplicates(yes_image_list, path_yes)

        #då ovan dict innehåller både både "orginal" och kopior så sorterar vi dem en spara lista och en delete lista
        no_keep, no_delete = self.which_files_delete(get_no_duplicates)
        yes_keep, yes_delete = self.which_files_delete(get_yes_duplicates)

        #skapar mappar en för kopior och en ny mapp utan dubbletter
        duplicate_folder_no, new_no_folder = self.create_folders(data_path, "no")
        duplicate_folder_yes, new_yes_folder = self.create_folders(data_path, "yes")

        #icke-snygg pythonkod för att spara filer i rätt mapp, kopior och icke-kopior
        self.move_and_delete_dup(duplicate_folder_no, no_delete)
        self.move_and_delete_dup(new_no_folder, no_keep)
        self.move_and_delete_dup(new_no_folder, no_without_dup)
        self.move_and_delete_dup(duplicate_folder_yes, yes_delete)
        self.move_and_delete_dup(new_yes_folder, yes_keep)
        self.move_and_delete_dup(new_yes_folder, yes_without_dup)

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
        training_folder_paths = self.checker(list_of_folders, '')

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

    def check_duplicates(self, image_list, path):
        """In lista med bilder och ut en dict med duplicerade bilder"""
        list_of_pixel = []                  #lista totalt antal pixlar per bild
        unique_pixel = set()                #lista unika pixel värden
        list_of_duplicate_pixel = []        #lista duplicerade pixel värden
        dict_duplicates = {}                #dictionary av de duplicerade värdena(value) och path (key)
        list_of_pic_no_dup = []             #lista på alla bilder som inte har någon kopia

        #för varje bild i image_list lägg till summerade pixelvärdet i list_of_pixel    
        for i in range(len(image_list)):
            list_of_pixel.append(np.sum(image_list[i]))
        
        #för varje pixelvärde i list_of_pixel lägg unika värden i unique_pixel och duplicerade i duplicate_pixel_pos
        for num in list_of_pixel:
            if num not in unique_pixel:
                unique_pixel.add(num)
            else:
                unique_pixel = unique_pixel - set([num])
                list_of_duplicate_pixel.append(num)
        
        #för varje pixelvärde i image_list som finns i duplicated_pixel_pos spara path (key) och pixelvärde (value) i en dict
        for i in range(len(image_list)):
            if np.sum(image_list[i]) in list_of_duplicate_pixel:
                dict_duplicates[path[i]] = os.stat(path[i]).st_size
        
        dict_sorted = self.sortera_dict(dict_duplicates)    #kör funktionen för att sortera dict
        
        #sparar här ned alla bildsökvägar som inte finns med i dict_sorted
        for i in path:
            if i not in dict_sorted:
                list_of_pic_no_dup.append(i)
        return dict_sorted, list_of_pic_no_dup

    def sortera_dict(self, duplicates):
        """Får in ett dict och skickar ut ett sorterat dict"""
        sorted_dict = {}
        sorted_keys = sorted(duplicates, key=duplicates.get)  
        for w in sorted_keys:
            sorted_dict[w] = duplicates[w]
        return sorted_dict

    def images_to_list(self, path):
        image_list = []
        #spara varje bild som en numpy array i en lista
        for path in path:
            img = cv2.imread(str(path))
            image_list.append(img)
        return image_list

    def which_files_delete(self, sorted_dict):
        """In vår dict med dupplicerade värden, ut 2 listor, en med unika värden och en med de filer som skall tas bort. """
        listan = []
        listan_pixel = []
        listan_delete = []
        
        for key in sorted_dict:
            if sorted_dict[key] not in listan_pixel:
                listan_pixel.append(sorted_dict[key])
                listan.append(key)
            else:
                listan_delete.append(key) 
        print(f"Filer att spara: {len(listan)}")
        print(f"Filer att ta bort: {len(listan_delete)}")
        return listan, listan_delete

    def create_folders(self, data_path, word):
        duplicate_folder = Path(data_path / "duplicate_folder" / word)
        without_dup_folder = Path(data_path / "without duplicates" / word)
        if not exists(duplicate_folder):
            os.makedirs(duplicate_folder)
        if not exists(without_dup_folder):
            os.makedirs(without_dup_folder) 
        return duplicate_folder, without_dup_folder 

    def move_and_delete_dup(self, to_folder, list_of_images):
        """funktionen skapar en kopia av bilder som skall tas bort och flyttar dessa till en mapp
        sedan tas dubbletter bort från originalmappen"""
        for file in list_of_images:   
            filename = str(Path(to_folder / file.name))     #path och filnamn och filtyp
            img = cv2.imread(str(file))                     #läs bild         
            cv2.imwrite(filename, img)                      #spara dubblett
        
