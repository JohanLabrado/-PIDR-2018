import numpy as np
import cv2
import math
import copy
import time
import csv
from xml.dom.minidom import *
import matplotlib.pyplot as plt
import colorsys
import sys
import datetime

cap = cv2.VideoCapture('C:\\Users\\Johan\\Desktop\\TELECOM\\2A\\PIDR\\-PIDR-2018\\PIDR - 2018\\Videos\\BC_MESS_1_1.webm',0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=50000, varThreshold=50, detectShadows=False) #Changement de la valeur du seuil en jouant sur varThreshold

def background_subtraction_function(frame):
    # Background substraction
    kernel1 = np.ones((2, 2), np.uint8)  # opening
    kernel2 = np.ones((9, 9), np.uint8)  # closing
    kernel3 = np.ones((3, 3), np.uint8)  # opening
    # nombre impair pour symétrie (9 ou 11)
    a = fgbg.apply(frame)
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN,  kernel1)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel2)
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN,  kernel3)

    final_img = cv2.bitwise_and(frame, frame, mask=a)
    thresh, final_img2 = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return final_img, final_img2

def draw(event,x,y,flags,param):

    if event == 0:
        global xMouse
        global yMouse
        xMouse,yMouse = x,y

        #print xMouse,yMouse

def compareCouleur(x,y,w,h,compteur):
    dst = cv2.inRange(frameHSV[y:y+h,x:x+w], equipe1-[erreurDeg,erreurRay,luminance], equipe1+[erreurDeg,erreurRay,luminance])
    nb1 = cv2.countNonZero(dst)
    dst = cv2.inRange(frameHSV[y:y+h,x:x+w], equipe2-[erreurDeg,erreurRay,255], equipe2+[erreurDeg,erreurRay,255])
    nb2 = cv2.countNonZero(dst)
    dst = cv2.inRange(frameHSV[y:y+h,x:x+w], equipe3-[erreurDeg,erreurRay,luminance], equipe3+[erreurDeg,erreurRay,luminance])
    nb3 = cv2.countNonZero(dst)

    comp,distance,xretour,yretour = compareCentre(x+w/2, y+h/2, compteur)
    cv2.circle(frameAffiche,(x+w/2,y+h/2),8,(50,255,50),-1)

    if nb3>(w*h)/facteur:
        coul = "arbitre"
    elif nb1>(w*h)/facteur:
        coul = "red"
    elif nb2>(w*h)/facteur:
        coul = "white"
    else:
        coul = "none"
    message = compte(coul,comp)
    cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (0,0,0), 2)




#################################
ret1, framePrec = cap.read()
(n,m,z) = framePrec.shape
grayPrec1 = np.ones((n,m))*255
grayPrec2 = np.ones((n,m))*255

x = 0
xMouse=0
yMouse=0
positionsLFG=[]
positionsBG=[]
positionsLFD=[]
positionsBD=[]
isOnScreen = True



listeSommets = []

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',draw)
ret, frame = cap.read()





emplacements=["haut a gauche","bas a droite"]

#zone gauche

for i in range (0,2) :  # Zone lancer franc gauche
    ret, frame = cap.read()
    messagePosition="Positionnez la souris sur le coin en "+emplacements[i]+" de la zone de lancer franc gauche puis appuyez sur Espace pour valider, ou sur une autre touche pour quitter"
    cv2.putText(frame, messagePosition, (432, 150), 0, 0.35, (0, 0, 255))
    if i == 1 :
        cv2.circle(frame, (positionsLFG[0][0], positionsLFG[0][1]), 5, (0, 0, 100), -1)
    cv2.imshow('frame', frame)
    waitedSpace = cv2.waitKey(0)
    if waitedSpace == 32 :
        positionsLFG.append((xMouse,yMouse))
        #cv2.imshow('frame', frame)
        print(positionsLFG)

cv2.rectangle(frame, positionsLFG[0], positionsLFG[1], (0, 0, 255), 5)

for i in range (0,2) :
    ret, frame = cap.read()
    cv2.rectangle(frame, positionsLFG[0], positionsLFG[1], (0, 0, 255), 5)
    messagePosition="Positionnez la souris sur le coin en "+emplacements[i]+" de la bouteille gauche puis appuyez sur Espace pour valider, ou sur une autre touche pour quitter"
    cv2.putText(frame, messagePosition, (432, 150), 0, 0.35, (0, 0, 255))
    if i == 1 :
        cv2.circle(frame, (positionsBG[0][0], positionsBG[0][1]), 5, (0, 0, 100), -1)
    cv2.imshow('frame', frame)
    waitedSpace = cv2.waitKey(0)
    if waitedSpace == 32 :
        positionsBG.append((xMouse,yMouse))
        #cv2.imshow('frame', frame)
        print(positionsBG)

##zone droite

#création de la zone de lancer franc droit
for i in range (0,2) :  # Zone lancer franc droit
    ret, frame = cap.read()
    cv2.rectangle(frame, positionsLFG[0], positionsLFG[1], (0, 0, 255), 5)
    cv2.rectangle(frame, positionsBG[0], positionsBG[1], (0, 255, 0), 5)
    messagePosition="Positionnez la souris sur le coin en "+emplacements[i]+" de la zone de lancer franc droit puis appuyez sur Espace pour valider, ou sur une autre touche pour quitter"
    cv2.putText(frame, messagePosition, (432, 150), 0, 0.35, (0, 0, 255))
    if i == 1 :
        cv2.circle(frame, (positionsLFD[0][0], positionsLFD[0][1]), 5, (0, 0, 100), -1)
    cv2.imshow('frame', frame)
    waitedSpace = cv2.waitKey(0)
    if waitedSpace == 32 :
        positionsLFD.append((xMouse,yMouse))
        #cv2.imshow('frame', frame)
        print(positionsLFD)

cv2.rectangle(frame, positionsLFD[0], positionsLFD[1], (0, 0, 255), 5)

#création de la zone de la bouteille de droite
for i in range (0,2) :
    ret, frame = cap.read()
    cv2.rectangle(frame, positionsLFG[0], positionsLFG[1], (0, 0, 255), 5)
    cv2.rectangle(frame, positionsBG[0], positionsBG[1], (0, 255, 0), 5)
    cv2.rectangle(frame, positionsLFD[0], positionsLFD[1], (0, 0, 255), 5)
    messagePosition="Positionnez la souris sur le coin en "+emplacements[i]+" de la bouteille droite puis appuyez sur Espace pour valider, ou sur une autre touche pour quitter"
    cv2.putText(frame, messagePosition, (432, 150), 0, 0.35, (0, 0, 255))
    if i == 1 :
        cv2.circle(frame, (positionsBD[0][0], positionsBD[0][1]), 5, (0, 0, 100), -1)
    cv2.imshow('frame', frame)
    waitedSpace = cv2.waitKey(0)
    if waitedSpace == 32 :
        positionsBD.append((xMouse,yMouse))
        #cv2.imshow('frame', frame)
        print(positionsBD)

##############""""""""""""""""""""""""""""""""""

ret,frame_4 = cap.read()
ret,frame_3 = cap.read()
ret,frame_2 = cap.read()
ret,frame_1 = cap.read()

################################

ret1, framePrec = cap.read()
(n,m,z) = framePrec.shape
grayPrec1 = np.ones((n,m))*255
grayPrec2 = np.ones((n,m))*255
erreur = 50
compteur = 0
rougeOk = 0
rougePasOk = 0



x = 0
xMouse=0
yMouse=0
isOnScreen = True

path = 'C:\\Users\\Johan\\Desktop\\TELECOM\\2A\\PIDR\\-PIDR-2018\\PIDR - 2018\\Résultat\\'
output_path = path + 'test.csv'


liste = []
joueur=[]

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',draw)
ret, frame = cap.read()
cv2.imshow('frame',frame)
cv2.rectangle(frame, positionsLFG[0], positionsLFG[1], (0, 0, 255), 5)
cv2.rectangle(frame, positionsBG[0], positionsBG[1], (0, 255, 0), 5)
cv2.rectangle(frame, positionsLFD[0], positionsLFD[1], (0, 0, 255), 5)
cv2.rectangle(frame, positionsBD[0], positionsBD[1], (0, 255, 0), 5)
timer = 0.001
compteur = 0
stop = False

with open(output_path, 'w', newline='\n') as csvfile:
    file_writer = csv.writer(csvfile, delimiter=';')
    # file_reader = csv.reader(csvfile) # Reader
    file_writer.writerow(['x', 'y', 'color'])
    compteurFrameLFG =  0 # Compteur de frames en situation de lancer franc à gauche.
    compteurFrameLFD =  0 # Compteur de frames en situation de lancer franc à droite.
    lastZeroG = 0  # Dernière frame de remise à zéro, signifiant le début de la situation de lancer franc à gauche.
    lastZeroD = 0  # Dernière frame de remise à zéro, signifiant le début de la situation de lancer franc à droite.
    while(cap.isOpened() ):
        ret, frame = cap.read()
        cropped_img, frame2 = background_subtraction_function(frame)
        frequence = 1/timer
        message = "frame : %d; img/s : %d, time : %d mn %f s" %(compteur,frequence,(compteur/25)//60,(compteur/25)%60)

        joueur=[]
        #frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray1 = cv2.cvtColor(frame_4, cv2.COLOR_BGR2GRAY)
        #frame2 = cv2.absdiff(gray, gray1)
        ret, grayOuvert = cv2.threshold(frame2, 127, 255, cv2.THRESH_TOZERO)
        frameAffiche = copy.deepcopy(frame)
        im2, contours, hierarchy = cv2.findContours(grayOuvert.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frameAffiche = copy.deepcopy(frame)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if h > 20 and w > 20:
                joueur.append([x + w / 2, y + h / 2, w * h])
                print(x+h,y+w)
                cv2.rectangle(frameAffiche, (x, y), (x + w, y + h), (255, 0, 0), 1)
                #if compteur > 1 and compteur <1000 :

                cv2.circle(frameAffiche, (int(math.floor(x+w/2)), int(math.floor(y+h/2))),1, (0, 0, 255), 5)


        file_writer.writerow([compteur])
        compteurJoueurLFG = 0  # Nombre de joueur en situtation de lancer franc à gauche dans la frame en cours
        compteurJoueurLFD = 0  # Nombre de joueur en situtation de lancer franc à droite dans la frame en cours

        for j in joueur:
            lf = ""
            xr = int(j[0])
            yr = int(j[1])
            if xr<=positionsLFG[1][0] and xr>=positionsLFG[0][0] and yr<=positionsLFG[1][1] and yr>=positionsLFG[0][1] :#and (xr<=positionsBG[0][0] or xr >= positionsBG[1][0]) and (yr <=positionsBG[1][0] or yr >= positionsBG[1][1]):  # Si on est dans le zone
                lf="Lancer franc à gauche"
                compteurJoueurLFG += 1
                if compteur > 500 and compteur < 1000:
                    cv2.circle(frame, (xr, yr), 5, (0, 0, 100), -1)
                    print("ttttttttttttttttttttttt")
                    print(xr,yr)
            if xr<=positionsLFD[1][0] and xr>=positionsLFD[0][0] and yr<=positionsLFD[1][1] and yr>=positionsLFD[0][1] :#and (xr<=positionsBD[0][0] or xr >= positionsBD[1][0]) and (yr <=positionsBD[1][0] or yr >= positionsBD[1][1]):  # Si on est dans le zone
                lf="Lancer franc à droite"
                compteurJoueurLFD += 1
            file_writer.writerow([xr,yr,lf])  # Génération CSV
            #if color=="red":
            #    cv2.circle(frame,(xr,yr),15,(0,0,100),-1)
            #else:
            #   cv2.circle(frame,(xr,yr),15,(0,100,0),-1)

        if compteurJoueurLFG != 1 :
            compteurFrameLFG = 0
            if compteur-lastZeroG > 40 :  # On regarde le temps qu'a duré la situation de lancer franc.
                file_writer.writerow(["Fin de lancer franc à gauche, le debut etait à la frame", lastZeroG, "Temps correspondant", lastZeroG/25 ])
            lastZeroG=compteur
        else :
            compteurFrameLFG += 1

        if compteurJoueurLFD != 1 :
            compteurFrameLFD = 0
            if compteur-lastZeroD > 40 :  # On regarde le temps qu'a duré la situation de lancer franc.
                file_writer.writerow(["Fin de lancer franc à droite, le debut etait à la frame", lastZeroD, "Temps correspondant", lastZeroD/25 ])
            lastZeroD=compteur
        else :
            compteurFrameLFD += 1
        if isOnScreen :
            cv2.putText(frameAffiche,message,(25,25),0,0.7,(0,0,255),)
        else:
            cv2.putText(frameAffiche,message,(25,25),0,0.7,(0,125,0))
        #cv2.imshow('frame',frame)
        k=cv2.waitKey(1)
        if k == 32:
            isOnScreen = not(isOnScreen)
        elif k == 65365: #touche u
            timer = timer/2
        elif k ==65366:  #touche v
            timer = 2*timer
        elif k==10:
            cv2.putText(frame,"Pause",(250,250),0,8,(125,0,0),4)
            cv2.imshow('frame',frame)
            space = cv2.waitKey(0)
            while( space != 10):
                space = cv2.waitKey(0)
        elif k == ord('z'):
            stop = True
            break
        elif k == ord('q'):
            break

        #cv2.imshow('frame',frame)

        if isOnScreen:
            liste.append((xMouse,yMouse))
            cv2.circle(frame,(xMouse,yMouse),5,(255,0,0),-1)
            cv2.rectangle(frameAffiche, positionsLFG[0], positionsLFG[1], (0,0,255), 5)
            cv2.rectangle(frameAffiche, positionsBG[0], positionsBG[1], (0, 255, 0), 5)
            cv2.rectangle(frameAffiche, positionsLFD[0], positionsLFD[1], (0, 0, 255), 5)
            cv2.rectangle(frameAffiche, positionsBD[0], positionsBD[1], (0, 255, 0), 5)


        else :
            liste.append((-1,-1))
            cv2.rectangle(frameAffiche, positionsLFG[0], positionsLFG[1], (0, 0, 255), 5)
            cv2.rectangle(frameAffiche, positionsBG[0], positionsBG[1], (0, 255, 0), 5)
            cv2.rectangle(frameAffiche, positionsLFD[0], positionsLFD[1], (0, 0, 255), 5)
            cv2.rectangle(frameAffiche, positionsBD[0], positionsBD[1], (0, 255, 0), 5)

        compteur = compteur +1
        cv2.imshow('frame',frameAffiche)
        time.sleep(timer)


cap.release()
cv2.destroyAllWindows()
compteur = 0



#coordonnes :
# haut gauche : 92 : 289
#haut droit  : 173 : 289
#bas droit :  173 : 425
#bas gauche : 92 : 425