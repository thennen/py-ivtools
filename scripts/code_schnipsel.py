######### erlaubt es alle daten nach namen zu sortieren, sodass das erstelldatum passt ############

os.chdir('X:/emrl/Pool/Bulletin/Witzleben/Messungen/RF/Rohdaten/RF_Messplatz/AH2/x10y08_RESET/sortiert/0dB/')
files = os.listdir()
for f in files:
    if len(f) == 10:
        filename= f[:5] + '0' + f[5:]
        print(filename)
        os.rename(f, filename)