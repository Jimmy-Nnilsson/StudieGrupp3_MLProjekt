import numpy as np
import cv2

class BEX_cropping():
    """Cropping class som tar emot fullständig sökväg, cropppar enligt algoritm o returnerar np-array"""

    def __init__(self, f_name: str):
        self.file_name = f_name


    def calculate_cropping(self):
        frame = 3                                             # klipper av en ram runt bilden direkt eftersom mkt smuts sitter där
        IMG_SIZE = (224, 224)                                    # detta är storleken som bilderna rezas till
        shape_comparison = []
        fname = self.file_name
        image = cv2.imread(str(fname))                          
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        temp = gray[frame:-frame, frame:-frame]             # temp är bild-numpyn
        res_std_x = np.std(temp, axis = 0)                  #skapar en arr med standardavvikelse för x och y
        res_std_y = np.std(temp, axis = 1)

        res_std_x = self.encode(res_std_x, 0.1)                  # encode gör en arr "binär"; allt < 0.1 -> 0 allt annat = 1
        res_std_y = self.encode(res_std_y, 0.1)

        calc_arr = self.encode(temp, 40, 255)                    # ändrar tröskelvärdet i bilden. Allt under 40 -> 0, allt annat = 255

        res_sum_x = np.sum(calc_arr, axis = 0)              # skapar en arr med summor av linjer i x och y
        res_sum_y = np.sum(calc_arr, axis = 1)

        res_sum_x = self.encode(res_sum_x, 1000)                  # gör binär kodning; allt < 1000 -> 0, allt annat =1
        res_sum_y = self.encode(res_sum_y, 1000)

        sigma_x = res_std_x * res_sum_x                     # multliplicerar ihop de båda metoderna sum * std
        sigma_y = res_std_y * res_sum_y                     # då framträder det 0 på alla rader med imformation som ska bort

        half_width = int(.5 * len(sigma_x))                 # delar bildens bredd för sökning
        half_height = int(.5 * len(sigma_y))                # delar bildens höjd för sökning

        left_val = np.where(sigma_x[:half_width] == 0)[0]       # söker från vänster mot mitten
        right_val = np.where(sigma_x[half_width:] == 0)[0]      # söker från mitten till högerkant

        top_val = np.where(sigma_y[:half_height] == 0)[0]       # söker från toppen
        bottom_val = np.where(sigma_y[half_height:] == 0)[0]    # söker från mitten ner mot botten


        if len(left_val) > 0:                       # om värde '0' hittats så är längden > 0
            left_x = left_val[-1]                   # tar sista värdet i arr dvs det närmst mitten
        else:
            left_x = 0 

        if len(right_val) > 0:                      # om värde '0' hittats så är längden > 0
            right_x = right_val[0] + half_width     # tar först värdet i arr eftersom sökningen börjar mitt i bild
        else:
            right_x = len(sigma_x)

        if len(top_val) > 0:                        # om värde '0' hittats så är längden > 0
            top_y = top_val[-1]                     # tar sista värdet i arr dvs det närmst mitten
        else:
            top_y = 0 

        if len(bottom_val) > 0:                     # om värde '0' hittats så är längden > 0
            bottom_y = bottom_val[0] + half_height  # tar först värdet i arr eftersom sökningen börjar mitt i bild
        else:
            bottom_y = len(sigma_y)

        new_image = gray[top_y + frame:bottom_y + frame, left_x + frame:right_x + frame]    # här appliceras frame på alla mått
        
        #----sparar undan shapes pre / post
        temp_shape = []
        temp_shape.append(image.shape[0:2])         # 0-2 för att skippa sista dim som bara anger lager i bild (RGB=3)
        temp_shape.append(new_image.shape)
        shape_comparison.append(temp_shape)         # sparar undanstorleken för att jämföra vilka bilde som beskärdes mest
        # -----------------------------------

        new_image = cv2.resize(new_image, IMG_SIZE)

        return new_image

#---------------hjälp funktion--------------------------------
    def encode(self, arr, thresh, max = 1):
        """" Tar emot en array, ersätter med 0 eller max-värde. Brytpunkten ligger vid värdet 'thresh' """

        arr = np.where(arr < thresh, 0, arr)
        arr = np.where(arr != 0, max, arr).astype(int)
        return arr


