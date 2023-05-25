import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
class dataReader():
    def __init__(self, locations=[], training=True):

        zones = gpd.read_file("ambulance-location-allocation\scripts\data\grid_zones.geojson")
        zones["geometry"] = zones["geometry"].to_crs(epsg=4326)

        zones["g_id"] = zones.index

        self.frame = pd.read_csv("./ambulance-location-allocation/scripts/proprietary_data/cleaned_data.csv")

        file = "./ambulance-location-allocation/scripts/data/grid_centroids.csv"
        centroid = pd.read_csv(file)
        self.frame = self.frame.merge(centroid, on=["xcoor", "ycoor"], how="left")
        self.frame = self.frame.dropna()


        self.frame = gpd.GeoDataFrame(
                            self.frame, geometry=gpd.points_from_xy(self.frame.long, self.frame.lat))
        self.frame.set_crs(epsg=4326, inplace=True)
        self.frame["geometry"] = self.frame["geometry"].to_crs(epsg=4326)
        self.frame = gpd.sjoin(self.frame, zones, how="left", op="within")
        self.frame = self.frame[self.frame["hastegrad"].isin(["A", "H"])]


        for f in ["tidspunkt", "tiltak_opprettet", "varslet", "rykker_ut", "avg_hentested", "ank_hentested", "ank_levsted", "ledig"]:
            self.frame[f] = pd.to_datetime(self.frame[f], errors="coerce")
        self.frame = self.frame[self.frame["ank_hentested"] > self.frame["rykker_ut"]]

        
        self.frame["time_used"] = pd.to_timedelta(self.frame["ank_hentested"]-self.frame["tidspunkt"], unit="s")
        self.frame["time_used"] = np.array(self.frame["time_used"], dtype='timedelta64[s]').astype("float")/60



        self.frame = self.frame.dropna()

        self.frame = self.frame[(self.frame["tidspunkt"].dt.year > 2005) & (self.frame["tidspunkt"].dt.year < 2019)]

        self.frame = self.frame.dropna()


        self.frame["night"] = (self.frame["tidspunkt"].dt.hour > 20) | (self.frame["tidspunkt"].dt.hour < 8)
       
        self.frame["g_id"] = self.frame["g_id"].astype("float64").astype("str")


        self.A, self.H = self.create_survival(self.frame)
        if len(locations) > 0:
            print(len(self.frame.index))
            self.frame = self.frame[self.frame["g_id"].isin(locations)]


        # IS GENERALLY CLEAN
    
        self.frame = self.frame.sort_values(by="tidspunkt")

        indexes = np.where(self.frame["night"] == False)[0][0]
        indexes += 1
        self.frame = self.frame.iloc[indexes:] # starts in day

        self.frame["t-d"] = self.frame["rykker_ut"]-self.frame["tidspunkt"]
        self.frame["i-d"] = self.frame["avg_hentested"]-self.frame["ank_hentested"]
        self.frame["h-l"] = self.frame["ledig"]-self.frame["ank_levsted"]
        for f in ["t-d", "i-d", "h-l"]:
            self.frame[f] = pd.to_timedelta(self.frame[f], unit="s")
            self.frame[f] = np.array(self.frame[f], dtype='timedelta64[s]').astype("float")/60/60
            self.frame = self.frame[self.frame[f] > 0]
            self.frame = self.frame[self.frame[f] < np.quantile(self.frame[f], 0.95)]
        
      
        self.frame["iTime"] = self.frame["tidspunkt"].diff().abs().astype("timedelta64[s]").shift(-1) # peek value
        self.frame = self.frame[self.frame["iTime"] > 0]
        self.frame["iTime"] = self.frame["iTime"].astype("timedelta64[s]")

        self.frame = self.frame[["hastegrad", "tidspunkt", "iTime", "g_id", "night", "lat", "long", "t-d", "i-d", "h-l"]]

        self.train, self.test = self.dataSplit(self.frame)
        self.stTime = np.datetime64(self.train["tidspunkt"].iloc[0])

        self.iter = 0
        self.epoch = 0
        self.day = True
        self.training = training

        #self.test = self.test[:100]
        self.row = None
        self.switch = False
        #self.test = self.test[:100]
        
 
    def create_survival(self, frame):
        for f in ["A", "H"]:
            test = frame[frame["hastegrad"] == f]
            a = test["time_used"]
            x = np.linspace(np.min(a), np.max(a), 1000)
            vals = []
            for i in x:
                vals.append(len(a[a>i])/len(a))
            if f == "A":
                self.A = [x, np.array(vals)]
            elif f == "H":
                self.H = [x, np.array(vals)]
        return self.A, self.H
    def get_survival(self, x, pri=0):
        """
        H : pri=0
        A : pri=1
        0: t-d, 1: i-d, 2: h-i
        """
        x2, y = None, None
        if pri == 0:
            x2 = self.H[0]
            y = self.H[1]
        else:
            x2 = self.A[0]
            y = self.A[1]
        ind = np.argmin(np.abs(x2-x))
        #print(y[ind], pri, x)
        return y[ind]

    def reset_iter(self):
        self.iter = 0
        self.epoch = 0
    def forward(self):
        if self.training:
            df = self.train
        else:
            df = self.test
        self.iter += 1
        
        if self.iter > (len(df.index)-2):
            self.epoch += 1
        
        if self.iter > len(df.index)-1:
            self.iter = 0
        try:
            row = df.iloc[self.iter]
        except:
            self.iter = 0
            row = df.iloc[self.iter]
        self.switch = False
        toNight = False
        #print(row["iTime"])
        #print(row["night"])
        if not ((not bool(row["night"])) == self.day):
            toNight = self.day
            self.day = not bool(row["night"])
            self.switch = True

        self.row = row
        #print(row["g_id"])

        return row, self.switch, toNight
    
    def dataSplit(self, df, test_size=0.33):
        size = len(df.index)
        splitpoint = round(size*(1-test_size))
        train = df.iloc[:splitpoint]
        test = df.iloc[splitpoint:]
        print(train)
        print(test)
        print(len(train.index), len(test.index))
        return train, test

#a = dataReader()

"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#X = np.array(df["processed"].tolist())
X = np.array(df[feat])
Y = df["dTime"]
test_size = 0.33
#splitpoint = round(len(df.index)*(1-test_size))
splitpoint = round(X.shape[0]*(1-test_size))

X_train, y_train = X[:splitpoint], Y[:splitpoint]
X_test, y_test = X[splitpoint:], Y[splitpoint:]

#splitpoint = round(len(X_test.index)*(1-test_size))
splitpoint = round(X_test.shape[0]*(1-test_size))
X_val, y_val = X_test[:splitpoint].copy(), y_test[:splitpoint].copy()
X_test, y_test = X_test[splitpoint:], y_test[splitpoint:]



"""