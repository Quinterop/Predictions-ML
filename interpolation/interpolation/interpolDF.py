import pandas as pd
import numpy as np
import warnings
import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import timedelta


def interpolation(X1,X2,Y1,Y2,XInconnu):
    #prevent division by 0
    if(X2 == X1):
        return Y1
    YInconnu = Y2*((XInconnu-X1)/(X2-X1))  +  Y1*((X2-XInconnu)/(X2-X1))
    return YInconnu

def df_interpolation(dataframe1, T_fld_name1, dataframe2, T_fld_name2, fld_name) :
    # on prend 2 df complets, on ajoute la colonne fld_name a dataframe1 pour les timings coincidants et on interpole les autres valeurs

    # dataframe1 : données réelles
    # T_fld_name1 : nom de la colonne de temps de dataframe1
    # dataframe2 : df avec les temps et les données de  fld_name à interpoler
    # T_fld_name2 : nom de la colonne de temps de dataframe2
    # fld_name : nom de la colonne à insérer dans dataframe1


    df = dataframe1.copy()
    df2 = dataframe2[[T_fld_name2, fld_name]].copy()
    
    print('DATAFRAMES')
    print(df)
    print(df2)

    if(fld_name in df.columns):
        return df.merge(df2, how='left', left_on=T_fld_name1, right_on=T_fld_name2)
    
    df[fld_name] = np.NaN
   
    if(type(df[T_fld_name1][0]) != type(df2[T_fld_name2][0])):
        print('error : different types of time')
        return

    # print('MERGED DFS')
    # print(df)

    df = df.sort_values(by=[T_fld_name1])
    df = df.reset_index(drop=True)
        
    #on insere les valeurs de df2 dans df si le temps correspond
    for row,index in df.iterrows():
        commonDates = df2.loc[df2[T_fld_name2] == df[T_fld_name1][row]]
        if(len(commonDates) == 0):
            continue
        mergeDate = commonDates.iloc[0]
        df[fld_name][row] = mergeDate[fld_name]
        # print('found matching date : ', df[T_fld_name1][row], 'value : ', df2date[fld_name])

    # print_full(df)
    
    #on interpole les valeurs manquantes

    for i in range(len(df)):
        if(np.isnan(df[fld_name][i])):
            #on récupère les lignes avant et après si elles existent
            line1 = None
            line2 = None

            for j in range(i+1, len(df)):
                if(not np.isnan(df[fld_name][j])):
                    line2 = df.iloc[j]
                    # print('line 2 found, value : ', df[fld_name][j])
                    break
                           
            for k in range(i-1, -1, -1):
                if(not np.isnan(df[fld_name][k])):
                    line1 = df.iloc[k]
                    # print('line 1 found, value : ', df[fld_name][k])
                    break

            #on interpole la ligne vide

                if(line1 is None and line2 is None):
                    print('erreur : pas de valeurs avant et après')
                if(line1 is None):
                    line1 = line2
                if(line2 is None):
                    line2 = line1
                    

            #conversion de la date
            if(type(line1[T_fld_name1]) == str):
                timeline1 = datetime.datetime.strptime(line1[T_fld_name1], '%Y-%m-%d %H:%M:%S')
            if(type(line2[T_fld_name1]) == str):
               timeline2 = datetime.datetime.strptime(line2[T_fld_name1], '%Y-%m-%d %H:%M:%S')
            if(type(df[T_fld_name1][i]) == str):
                timelinedf = datetime.datetime.strptime(df[T_fld_name1][i], '%Y-%m-%d %H:%M:%S')


            value = interpolation(timeline1, timeline2, line1[fld_name], line2[fld_name], timelinedf)

            df[fld_name][i] = value
            # print('line',i,'interpolated value :',value)
        else:
            pass
            # print('line',i,'is not empty')
            # print('value :',df[fld_name][i])
    return df    

def interpol_line(line1:pd.core.series.Series, line2:pd.core.series.Series, col2:str, target:int, toInterpol:list[str]=[], toNotInterpol:list[str]=[])->pd.core.series.Series:
    #col2 : colonne de temps 
    #target : temps de la nouvelle ligne
    #les temps doivent être des int ou float, peu importe l'échelle

    x1 = line1[col2]
    x2 = line2[col2]
    res = line1.copy()
    for col in res.index:
        res[col] = np.nan 
    res[col2] = target
    # interpolate all ints in the lines except col2
    for col in line1.index:
        if(col != col2  ):
            #check that col is numeric and not boolean
            if(pd.api.types.is_numeric_dtype(line1[col]) and  not pd.api.types.is_bool_dtype(line1[col])):
                if(col in toInterpol or col not in toNotInterpol):
                    y1 = line1[col]
                    y2 = line2[col]
                    print("interpolating", col, "from", y1, "to", y2, "for", target, "between", x1, "and", x2) 
                    print()
                    res[col]=(interpolation(x1, x2, y1, y2, target))
                    print("result", res[col])
    return res

def normalize(val, min, max):
    return (val-min)/(max-min)
    #FIX : floating point error

def denormalize(val, min, max):
    return val*(max-min)+min

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def interpolDf(df, timecol, pas_voulu, offset=0, toInterpol:list[str]=[], toNotInterpol:list[str]=[]):
    #la fonction recale un dataset sur une échelle de temps régulière en interpolant les valeurs 

    #df : dataframe à interpoler
    #timecol : colonne de temps
    #toInterpol : liste des colonnes à interpoler (facultatif)
    #toNotInterpol : liste des colonnes à ne pas interpoler (facultatif)
    #pas_voulu : pas voulu pour l'interpolation
    #offset : décalage à appliquer à la colonne de temps
    #les temps doivent être des int ou float, peu importe l'échelle
  
    #on trie le df par temps
    df = df.sort_values(by=timecol)

    #on récupère les temps min et max
    min = df[timecol].min()
    max = df[timecol].max() 

    #on normalise les temps et on arrondit a 5 chiffres après la virgule
    df[timecol] = df[timecol].apply(lambda x: normalize(x, min, max))
    df[timecol] = df[timecol].apply(lambda x: round(x, 5))

    #on crée une colonne original pour marquer les lignes originales et une colonne filled pour marquer les lignes interpolables
    df['original'] = True
    df['filled'] = True


    #on adapte le pas voulu et l'offset à la normalisation
    pas_voulu = pas_voulu/(max-min)
    offset = offset/(max-min)

    #on crée une liste de temps à interpoler arrondis à 5 chiffres après la virgule
    temps = np.arange(min + offset, 1, pas_voulu)
    temps = [round(t, 5) for t in temps]

    #on ajoute des lignes vides avec les temps à interpoler au df si elles n'existent pas déjà
    for t in temps:  
        if(t not in df[timecol].values): #warning penser a floating point error
            newline = pd.Series([np.nan for i in range(len(df.columns))], index=df.columns)
            newline[timecol] = t
            newline['original'] = False 
            newline['filled'] = False
            df = df.append(newline, ignore_index=True)
        
    
    #on trie le df par temps
    df = df.sort_values(by=timecol)
    df = df.reset_index(drop=True)
    print(df)
    #on interpole les lignes vides
    for i in range(len(df)):
        if(df['filled'][i] == False):
            #on récupère les lignes avant et après si elles existent
    
            line1 = None
            line2 = None

            #debug prints
            

            print()
            print('finding line 2')
            for j in range(i+1, len(df)):
                # print('j',j)
                # print('line j',df.iloc[j])
                if(df['filled'][j] == True):
                    line2 = df.iloc[j]
                    print('line 2 found')
                    break
                # else:
                #     print('line 2 NOT found')
                    
                    
            for k in range(i-1, -1, -1):
                # print('line at k',df.iloc[k])
                if(df['filled'][k] == True):
                    line1 = df.iloc[k]
                    print('line 1 found')
                    break
                # else:
                #     print('line 1 NOT found')
                    
                    

            #on interpole la ligne vide
            # print('LINE1',line1)
            # print('LINE2',line2)
            if(line1 is None or line2 is None):
                print("error : no line before or after line at", i)
                continue
            df.iloc[i] = interpol_line(line1, line2, timecol, df[timecol][i], toInterpol, toNotInterpol)
            df['filled'][i] = True


    #on dénormalise les temps
    df[timecol] = df[timecol].apply(lambda x: denormalize(x, min, max))

    #on supprime les lignes originales ?
    # df = df[~df['original']]

    #on supprime les colonne original et filled
    df = df.drop(columns=['original'])
    df = df.drop(columns=['filled'])


    return df


def df_generate_int(t_start , t_end, period, T_fld_name ='date_time') :
    # génère un dataframe vide avec un horodatage précis pour un temps compté en int
    times = []
    for t in range(t_start, t_end, period):
        times.append(t)
    df = pd.DataFrame(times, columns=[T_fld_name])
    return df

def df_generate_datetime(t_start , t_end, period, T_fld_name ='date_time') :
    # génère un dataframe vide avec un horodatage précis pour un temps compté en format datetime

    #on convertit tout en secondes
    start = int(t_start.timestamp())
    end = int(t_end.timestamp())
    step = int(period.total_seconds())

    times = []
    for t in range (start, end, step):
        times.append(t)

    # create an empty list to store the UTC datetime objects
    utc_times = []

    # convert the timestamps to datetime objects in UTC timezone
    for t in times:
        utc_times.append(pd.to_datetime(t, unit='s', utc=True))

    # create an empty list to store the Europe/Paris datetime objects
    paris_times = []

    # convert the UTC datetime objects to Europe/Paris timezone and remove the timezone information
    for t in utc_times:
        paris_time = t.astimezone(tz='Europe/Paris')
        paris_time = paris_time.replace(tzinfo=None)
        paris_times.append(paris_time)

    df = pd.DataFrame(paris_times, columns=[T_fld_name])
    return df