import networkx as nx
import numpy as np
import atexit
import pickle
import os

atIncidentTime = 10

# [[0,5], [3,0], [3,0],[0,5],[0,5]]
cache = {}
hcache = {}

count = 0
fileName = "cache.pickle"
fileName_2 = "hcache.pickle"
toLoad = False
toSave = False
if os.path.exists(fileName) and toLoad:
     with open(fileName, "rb") as f:
        cache = pickle.load(f)
if os.path.exists(fileName_2) and toLoad:
     with open(fileName_2, "rb") as f:
        hcache = pickle.load(f)
print(len(cache.keys()))
def save_cache():
    global cache, fileName, toSave, fileName_2
    if len(cache.keys()) > 0 and toSave:
        print("saving cache")
        with open(fileName, "wb") as f:
            pickle.dump(cache, f)
    if len(hcache.keys()) > 0 and toSave:
        print("saving cache")
        with open(fileName_2, "wb") as f:
            pickle.dump(hcache, f)
atexit.register(save_cache)

class ambulance():
    def __init__(self, gridLoc, statuses=["atBase", "tIncident", "aLoc", "tHospital", "aHospital", "tBase"]):
        self.gridLoc = gridLoc
        self.base_station = self.gridLoc
        self.onShift = True
        self.status = 0

        self.sleep = False # to set to -1 when arrive to base (aka unavailable)

        self.statusMap = statuses
        self.hospitalLocs = []

        self.incidentAssigned = None
        self.path = None
        self.available = False
        self.extratime = 0
        self.responseTime = -1
        self.inciPri = -1

        self.waitTimeStatus = [0,0,0]

        self.iter = 0
        
        self.attr = 0
        

    def heur(self, a, b):
        a, b = self.attr[a], self.attr[b]
        la1, lo2 = a
        la2, lo2 = b
        return haversine_np(la1, lo2, la2, lo2)
    def checkCache(self, f, t):
        global cache
        isThere = f in cache.keys()
        isThere2 = False
        if isThere:
            isThere2 = t in cache[f].keys()
            if isThere2:
                spath = cache[f][t]
                return True, True, spath
            return True, False, None
        return False, False, None
    def shortestPath(self, G, f, t):
        isThere, isThere2, spath = self.checkCache(f, t)
        if not (isThere2 and isThere):
            #spath = nx.shortest_path(G, str(self.gridLoc), str(incident.gridLoc), weight="min_time")
            spath = nx.astar_path(G, f, t, heuristic=self.heur, weight="min_time")
            if not isThere:
                cache[f] = {}
            cache[f][t] = spath
        return spath

    def assignIncident(self, G, time, incident):
        global cache, count
        if self.iter == 0:
            self.attr = nx.get_node_attributes(G, name="latLong")
        if (self.available or self.status in [0,1,5]) and self.onShift:
            self.available = False    
            self.status = 1
            self.incidentAssigned = incident
            self.inciPri = incident.pri

            spath = self.shortestPath(G, str(self.gridLoc), str(incident.gridLoc))
            
            self.path = path(G, spath, time)
            self.path.waitTime = self.waitTimeStatus[0]
        else:
            print(self.status, self.available)
            raise Exception("Ambulance not assignable")
        self.iter += 1
    def updateLoc(self, G, time):
        #print(time, self.status, self.gridLoc)
        if self.status > 0:
            self.gridLoc, done, atTimeFinish = self.path.get_pos(time, self.extratime)
            self.extratime = 0
            if atTimeFinish:
                self.extratime = self.path.extraTime
                self.available = True
                self.status += 1
                if self.status > len(self.statusMap)-1:
                    if not self.sleep:
                        self.status = 0
                    else:
                        self.status = -1
                        self.sleep = False
                    self.path = None

                elif self.status == 2: #done with incident, travel to hospital
                    #print(self.statusMap[self.status])
                    self.incidentAssigned.attended = True
                    self.incidentAssigned.update(time-self.extratime*np.timedelta64(1, "h")) # WHY CAN THIS BECOME <0
                    self.responseTime = self.incidentAssigned.wTime
                    self.responseTime = np.max([0, self.responseTime])

                    #print("ResponseTime: ", self.responseTime)
                    #print(self.incidentAssigned.wTime)
                    self.incidentAssigned = None

                    if self.gridLoc in hcache.keys():
                        spath = hcache[self.gridLoc]
                    else:
                        #print(dict.fromkeys(self.hospitalLocs), self.gridLoc)
                        length, spath = nx.multi_source_dijkstra(G, dict.fromkeys(self.hospitalLocs), self.gridLoc, weight="min_time")
                        spath.reverse()
                        hcache[self.gridLoc] = spath                    

                    self.path = path(G, spath, time)
                    self.path.waitTime = self.waitTimeStatus[1] # i-d

                    self.available = False
                    self.status = 3 # travveling to hospital
                elif self.status == 4: # finished at hospital
                    
                    self.status = 5
                    hpath = self.shortestPath(G, self.gridLoc, self.base_station)

                    self.path = path(G, hpath, time)
                    self.path.waitTime = self.waitTimeStatus[2] # h-l
                    self.available = True
                #print(self.status)
                return self.updateLoc(G, time)
            else:
                self.available = False
        return self.status, self.available, self.responseTime

    def canIncident(self, G, time):
        self.updateLoc(time)
        return (self.available or self.status in [0,1,5]) and self.onShift
    def reassign_base(self, bId):
        self.base_station = bId

class incident():
    def __init__(self, loc, pri, time):
        self.gridLoc = loc
        self.lalong = [0,0]
        self.pri = pri
        self.time = time.copy()
        self.wTime = 0
        self.attended = False
        self.statusW = []

    def update(self, time):
        if self.attended == True:
            self.wTime = time-self.time
            self.wTime = self.wTime.astype("timedelta64[s]").astype("float")/60
    def update_w(self, time):
        if (self.time > time):
            print(self.time, time)
            exit()
        return (time-self.time).astype("timedelta64[s]").astype("float")/60
import scipy
class wait_time_draw():
    """
    H : pri=0
    A : pri=1
    0: t-d, 1: i-d, 2: h-i
    """
    
    def __init__(self):
        self.df = pd.read_csv("transition_times_distr.csv")
        self.x = np.linspace(0,40, 1000)
        self.A = []
        self.H = []
        self.preProcess()

    def preProcess(self):
        for index, row in self.df.iterrows():
            p = tuple(eval(row["params"]))
            d = eval("scipy.stats."+row["distro"]+".pdf(self.x, *p)")
            d = d/np.sum(d)
            if row["hastegrad"] == "A":
                self.A.append(d)
            else:
                self.H.append(d)
    def get_w_times(self, pri):
        if type(pri) == str:
            pri = ["H", "A"].index(pri)
        ps = self.A if pri == 1 else self.H
        time = [] # fraction of hour
        for p in ps:
            time.append(np.random.choice(self.x, p=p)/60)

        #print(time)
        return time


class path():
    def __init__(self, G, path, time, waitTime=0, waitTimeFinish=0):
        self.timeChain = []

        self.start_time = time        
        self.waitTime = waitTime
        
        self.waitTimeFinish = waitTimeFinish
        self.extraTime = 0

        self.finishedIN = None

        self.path = path

        if len(path) > 0:
            self.fId = path[0]
            self.currentLoc = self.fId
            self.tId = path[-1]
            self.updatePath(G, path)

        

        
    def updatePath(self, G, path):
        subG = G.subgraph(path)
        #edge_labels = nx.get_edge_attributes(subG, 'min_time')
        #self.time_chain = [edge_labels.get(x, edge_labels.get((x[1],x[0]))) for x in zip(path[self.fId][self.tId], path[self.fId][self.tId][1:])]
        #self.time_chain = np.cumsum(self.time_chain)
        self.timeChain = [0]
        subGp = nx.path_graph(path)
        for ea in subGp.edges():
            self.timeChain.append(float(subG.edges[ea[0], ea[1]]["min_time"]))
        self.timeChain = np.array(self.timeChain, dtype="float")

        self.timeChain = np.cumsum(self.timeChain)


    def get_pos(self, time, extra_time=0): # returns currentlocation, isatfinish, waitTimeatfinishIsDone
        self.extraTime = 0
        if len(self.path) > 1:
            #print(self.start_time, time, extra_time)
            #print(time, self.start_time)
            time_passed = (time-self.start_time)
            #print(time, self.start_time, extra_time)
            time_passed = time_passed.astype("timedelta64[s]").astype("float")/60/60 +extra_time # convert to a fraction of hours
            #print(time_passed, self.timeChain[-1], self.waitTime)
            if self.waitTime != 0:
                if self.waitTime > time_passed:
                    self.finishedIN = (self.waitTime-time_passed) + self.timeChain[-1]
                    return self.fId, False, False
            if time_passed > self.waitTime:
                time_passed = time_passed-self.waitTime # time_passed is time used on driving-

            index = np.argmin(np.abs(self.timeChain-time_passed)) # interpolating where the ambulance is based on time since it started driving.
            if self.timeChain[index] > time_passed:
                index -= 1
            self.currentLoc = self.path[index]
            atLocTimeFinish = False

            #print(time_passed, self.timeChain[-1], index, len(self.timeChain), self.currentLoc, self.tId)

            if self.timeChain[index] < time_passed and self.currentLoc == self.tId:
                atLocTimeFinish = (time_passed-self.timeChain[index]) > self.waitTimeFinish
                self.extraTime = (time_passed-self.timeChain[index])-self.waitTimeFinish

            self.finishedIN = self.timeChain[-1]-time_passed
            return self.currentLoc, self.currentLoc == self.tId, atLocTimeFinish
        else:
            return self.currentLoc, True, True


"""
- incident occurs
- ambulance with closest euclidean distance is dispatched.
""" 
import pandas as pd
import time as time
import geopandas as gpd
import copy
import matplotlib.pyplot as plt

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

import copy
class simulator():
    def __init__(self):
        self.G = nx.read_edgelist("adjacencyFinalKnnSpeed3")
        
        self.locations = np.array(list(self.G.nodes))


        self.locMap = {}
        for i,l in enumerate(self.locations):
            self.locMap[l] = i
        self.locMap = np.vectorize(self.locMap.get)

        self.rates = self.prepareRates()

        self.timeSince = np.zeros(len(self.rates[self.rates > 0]))
        self.probs = []
        #self.timeSince = np.random.uniform(0.0, high=60.0, size=len(self.rates[self.rates > 0]))
        self.timeSinceMap = np.where(self.rates > 0)[0]

        self.zones = self.prepareZones()
        self.locXY = self.prepareLatLong()

        self.baseStation = self.prepareBaseData()
        self.hospitals = self.prepareHospitals()
        
        self.nAmbulance = int(self.baseStation["alloc"].sum())
        self.ambLoc = np.array(["0000.0"]*self.nAmbulance)

        self.ambStatus = np.zeros(self.nAmbulance)

        self.ambulances = self.prepareAmbulances()
        self.waitTimes = wait_time_draw()
        self.toWait = True


        self.ambulances_dist_back = []

        self.responseTime = []
        

        self.samples = 2000
        self.iter = 1
        

        self.startTime = np.datetime64('now')
        self.pAcute = 0.52832512 # fromData, 6.9 not acute
        self.lamb = 0.21
        self.incidents = []
        self.uniformLoc = False

    def prepareRates(self):
        rate = pd.read_csv("lambdaLocR.csv")
        rate["g_id_right"] = rate["g_id_right"].astype("str")
        rate = rate.set_index('g_id_right')['lamb']
        rate = list(map(rate.get, self.locations))
        rate = [-1 if v is None else v for v in rate]
        return np.array(rate)

    def prepareZones(self):
        zones = gpd.read_file("ambulance-location-allocation\scripts\data\grid_zones.geojson")
        zones["geometry"] = zones["geometry"].to_crs(epsg=4326)
        zones = zones[zones.index.isin(np.array(self.locations, dtype="float64"))]
        zones["geometry"] = zones["geometry"].centroid
        return zones

    def prepareLatLong(self):
        df = self.zones.copy()
        df["lat"] = df["geometry"].y
        df["long"] = df["geometry"].x

        df["strInd"] = df.index.astype("float64").astype("str")
        la = df.set_index('strInd')["lat"]
        lo = df.set_index('strInd')["long"]
        lal = list(map(la.get, self.locations))
        lol = list(map(lo.get, self.locations))
        d = {l : [lal[j], lol[j]] for j, l in enumerate(self.locations)}
        #d =  {l : [lol[j], lal[j]] for j, l in enumerate(self.locations)}
        nx.set_node_attributes(self.G, d, name="latLong")
        pos=nx.get_node_attributes(self.G,'latLong')
        nx.write_gpickle(self.G, "test.gpickle")
        #nx.draw(self.G, pos, node_size=2)
        #plt.show()

        res = [ [x,y] for x,y in zip(lal, lol)]
        return np.asarray(res)
    


    def keep_dist(self, loc, r): # r=km, loc=[lat, long]
        d = np.vectorize(haversine_np)
        dists = d(self.locXY[:, 1], self.locXY[:, 0], loc[1], loc[0])

        indexes = np.where(dists < r)[0]
        print(indexes)
        print("N locations: ", indexes.shape[0])
        print("% kept: ",  indexes.shape[0]/dists.shape[0])
        assert indexes.shape[0] > 0
        
        toKeep = self.locations[indexes].copy()
        
        self.G = self.G.subgraph(toKeep).copy()
        subg = [self.G.subgraph(c) for c in nx.connected_components(self.G)]
        i = np.array([len(s.nodes) for s in subg])
        toKeep = np.array(subg[np.argmax(i)].nodes)
        indexes = np.where(np.isin(self.locations, toKeep))[0].copy()

        print("N locations revised: ", indexes.shape[0])
        print("% kept revised: ",  indexes.shape[0]/dists.shape[0])


        self.G = self.G.subgraph(toKeep).copy()
        self.locations = toKeep.copy()


        self.rates = self.rates[indexes]
        self.timeSince = np.zeros(len(self.rates[self.rates > 0]))
        self.timeSinceMap = np.where(self.rates > 0)[0]

        toKeepAmb = np.isin(self.ambLoc, toKeep)
        self.ambLoc = self.ambLoc[toKeepAmb]

        self.ambulances = [self.ambulances[i] for i in np.where(toKeepAmb)[0]]
       
        self.nAmbulance = len(self.ambulances)
        print("N ambulances: ", self.nAmbulance)
        print("N base stations: ", np.unique(self.ambLoc).shape[0])

        self.ambStatus = np.zeros(self.nAmbulance)

        h = np.array(self.hospitals["index_right"])
        toKeepH = []
        for hos in h:
            if str(hos) in toKeep and self.G.has_node(str(hos)):
                toKeepH.append(hos)
        h = np.array(toKeepH)
        print(h)
        
        print("N hospitals: ", h.shape[0])
        assert h.shape[0] > 0
        for i in range(len(self.ambulances)):
            ambulance = self.ambulances[i]
            ambulance.hospitalLocs = h
            self.ambulances[i] = ambulance

        self.ambulances_dist_back = copy.deepcopy(self.ambulances)

        self.locXY = self.locXY[indexes]
        self.locMap = {}
        for i,l in enumerate(self.locations):
            self.locMap[l] = i
        self.locMap = np.vectorize(self.locMap.get)
        print(self.locations)
        toKeep = []
        for id in self.baseStation["index_right"]:
            if str(id) in self.locations:
                toKeep.append(id)
        #self.baseStation["dist"] = d(self.baseStation["longitude"], self.baseStation["latitude"], loc[1], loc[0])
        #self.baseStation = self.baseStation[self.baseStation["dist"] < r]
        self.baseStation = self.baseStation[self.baseStation["index_right"].isin(toKeep)]
        print(self.baseStation)

    def prepareBaseData(self, alloc=[1, 1, 1, 1, 3, 1, 3, 5, 4, 2, 2, 3, 2, 2, 3, 3, 3, 2, 3], alloc_night=[1, 1, 1, 0, 2, 0, 2, 3, 3, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2]):
        try:
            file = "./ambulance-location-allocation/scripts/data/base_stations.csv"
            base = pd.read_csv(file)
        except:
            file = "./ambulance-location-allocation/src/main/resources/base_stations.csv"
            base = pd.read_csv(file)

        print(alloc)
        base["alloc"] = alloc
        base["alloc"] = base["alloc"].astype("float")

        base["alloc_n"] = alloc_night
        base["alloc_n"] = base["alloc_n"].astype("float")

        base["alloc_diff"] = base["alloc"]-base["alloc_n"]
        print(base)
        #exit()
        base = gpd.GeoDataFrame(
            base, geometry=gpd.points_from_xy(base.longitude, base.latitude))
        base.set_crs(epsg=4326, inplace=True)
        #base = gpd.sjoin(base, self.zones, how="left", op="within")
        base = gpd.sjoin_nearest(base, self.zones, how="left")
        base["index_right"] = base["index_right"].astype("float").astype("str")
        print(np.array(base["index_right"]))

        return base
    def prepareHospitals(self):
        try:
            path = "C:/Users/jon39/Documents/ambulance-location-allocation/ambulance-location-allocation/src/main/resources/no/ntnu/ambulanceallocation/simulation/hospitals.csv"
            df = pd.read_csv(path, header=None)
        except:
            path = "./ambulance-location-allocation/src/main/resources/no/ntnu/ambulanceallocation/simulation/hospitals.csv"
            df = pd.read_csv(path, header=None)
        df.columns = ["name", "lat", "lon"]
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat))
        df.set_crs(epsg=4326, inplace=True)
        #df = gpd.sjoin(df, self.zones, how="left", op="within")
        df = gpd.sjoin_nearest(df, self.zones, how="left")
        df["index_right"] = df["index_right"].astype("float").astype("str")
        return df

    def prepareAmbulances(self):
        idAlloc = np.array(self.baseStation[["index_right","alloc"]])
        ambulances = []

        i = 0
        for base in idAlloc:
            id, n = base
            for z in range(int(n)):
                amb = ambulance(id)
                amb.hospitalLocs = np.array(self.hospitals["index_right"]).copy()
                ambulances.append(amb)
                self.ambLoc[i] = copy.copy(id)
                self.ambStatus[i] = 0
                i += 1
        return ambulances
    def switch_allocation(self):
        idAlloc = np.array(self.baseStation[["index_right","alloc", "alloc_n", "alloc_diff"]])
        i = 0
        z = 0
        #statuses = np.array([a.status for a in self.ambulances])
        
        #didFix = False
        for base in idAlloc:
            id, n, n_n, goal = base
            z = i
            n = int(n)
            #print(statuses[i:i+n], n, goal)
            for g in range(int(n)):
                a = self.ambulances[i]
                if a.status == 0 and goal > 0:
                    self.ambStatus[i] = -1
                    a.status = -1
                    goal -= 1
                    #print(a.base_station)
                i += 1
           
            if goal > 0:
                #print("missed goal")
                done = []
                amb = 0
                while goal > 0:
                    amb = np.random.choice(np.arange(int(n)))
                    amb += z
                    while amb in done:
                        amb = np.random.choice(np.arange(int(n)))
                        amb += z
                    if self.ambulances[amb].status > 0:
                        self.ambulances[amb].sleep = True
                        done.append(amb)
                        goal -= 1
                    #print(a.base_station, self.ambulances[amb].status)
                    
                #print("done:", done)

        #statuses = np.array([a.status for a in self.ambulances])
        #sleeps = np.array([a.sleep for a in self.ambulances])
        #print(statuses)
        #print(sleeps)


    def reset_allocation(self):
        for z in range(len(self.ambulances)):
            a = self.ambulances[z]
            if a.status == -1:
                a.status = 0
                self.ambStatus[z] = 0

            a.sleep = False
            self.ambulances[z] = a

    def get_location(self, t):
        self.timeSince += t.astype("float64")/60/60/24 # n days as float from seconds
        self.probs = 1-np.exp(-self.rates[self.rates > 0]*self.timeSince)
        self.probs = self.probs/self.probs.sum()
        self.probs = np.nan_to_num(self.probs, nan=1/len(self.probs))

        index = np.random.choice(np.arange(0, len(self.probs)), p=self.probs)
        self.timeSince[index] = 0
        return self.locations[self.timeSinceMap[index]]


    def dispatch(self, i, simulateTime, inci):
        amb = self.ambulances[i]
        amb.assignIncident(self.G, simulateTime, inci)
        if self.toWait:
            if not self.fromData:
                amb.waitTimeStatus = self.waitTimes.get_w_times(inci.pri)
            else:
                amb.waitTimeStatus = inci.statusW
        status, available, rTime = amb.updateLoc(self.G, simulateTime)
        self.ambStatus[i] = status
    

    def sendClosestByRoad_pred(self, inci, tStatus = [0,5]): # how to handle not available ambulances
        target = inci.gridLoc
        #ambBase = np.where(self.ambStatus == 0)[0] # only from base
        ambBase = np.where(np.isin(self.ambStatus, tStatus))[0] # from base, and from hospital
        ambBaseLoc = np.unique(self.ambLoc[ambBase]).tolist()
        # 520.0 not in G.. 

        length, spath = nx.multi_source_dijkstra(self.G, dict.fromkeys(ambBaseLoc), target, weight="min_time")
        ambLoc = spath[0]
        index = np.where((self.ambLoc == ambLoc) & (np.isin(self.ambStatus, tStatus)))[0][0]
        return index

import gym
from gym import spaces
from collections import deque
from dataReader import *

class dispatchEnv(gym.Env, simulator):
    def __init__(self, lamb, iters=100000, locState=True, LaLo=None, dist=None, waitTimes=False, fromData=True, training=True, surFunc=False): # updates at 2048
        #super(simulator).__init__()

        super(dispatchEnv, self).__init__()


        

        self.toWait = waitTimes

        self.LaLo = LaLo
        if(LaLo != None):
            self.keep_dist(LaLo, dist)
        self.dataReader = dataReader(self.locations, training)
        self.surFunc = surFunc

        if fromData:
            self.startTime = self.dataReader.stTime

        self.fromData = fromData
        

        self.training=training
        if training:
            self.responseTime = deque(maxlen=1000)
            if self.surFunc:
                self.surVals = deque(maxlen=1000)
        self.ambulances_dist_back = copy.deepcopy(self.ambulances)

        self.lamb = lamb
        self.locState = locState
        self.wrongCount = 0
        if locState:
            self.size = len(self.locations)
            if not self.surFunc:
                self.observation_space = spaces.Dict(
                    {
                        "ambulance": spaces.Box(0, self.nAmbulance, shape=(self.size,), dtype=int),
                        "incident": spaces.Box(0, 5, shape=(self.size,), dtype=int),
                        #"distmap" : spaces.Box(0, 1, shape=(self.size,), dtype=float)
                    }
                )
            else:
                self.observation_space = spaces.Dict(
                    {
                        "ambulance": spaces.Box(0, self.nAmbulance, shape=(self.size,), dtype=int),
                        "incident": spaces.Box(0, 5, shape=(self.size,), dtype=int),
                        "incident_sur": spaces.Box(0, 5, shape=(self.size,), dtype=float), # A mapping of survival function of this incident.
                        #"distmap" : spaces.Box(0, 1, shape=(self.size,), dtype=float)
                    }
                )
        else:
            self.size = self.nAmbulance
            
            self.observation_space = spaces.Dict(
                {
                #    "ambulance": spaces.Box(0, 1, shape=(self.size,), dtype=float),
                    "distmap" : spaces.Box(0, 1.5, shape=(self.size,), dtype=float)
                }
            )
        #self.observation_space = spaces.Box(0, 1, shape=(self.size,), dtype=float)
        
        self.action_space = spaces.MultiDiscrete([self.size, self.size])
        self.incidents = [self.gen_incident()]
        self.peek = None

        self.ambMap = []
        self.ambBaseLoc = []

        self.iters = iters
        self.currentIter = 0

    def get_iTime(self):
        if not self.fromData:
            iTime = np.random.exponential(1/self.lamb, size=1)#.astype("timedelta64[s]") # minutes
            iTime *= 60 # seconds
            iTime = iTime.astype("timedelta64[s]")[0]
            return iTime
        else:
            return self.dataReader.row["iTime"]
    
    def gen_incident(self, move=True, iTime=0, peek=None):
        if not self.fromData:
            if move:
                iTime = self.get_iTime()
                self.startTime = np.datetime64(self.startTime+iTime)

            location = super().get_location(iTime)
            urgency = np.random.choice([0,1], p=[0.57, 0.43])

            inci = incident(location, urgency, self.startTime)
            if peek != None:
                inci = incident(location, urgency, self.startTime+peek)
            #inci.lalong = self.locXY[np.isin(self.locations, location)].copy()
            inci.lalong = self.locXY[self.locMap(location)].copy()
        else:
            row, switch, day = self.dataReader.forward()
            if move:
                self.startTime = np.datetime64(row["tidspunkt"])

            #print(self.startTime)
            location = str(row["g_id"])
            #print(location)
            hGrad = row["hastegrad"]
            if hGrad == "H":
                hGrad = 0
            else:
                hGrad = 1

            #print(row[["t-d", "i-d", "h-l"]].to_list())
            if switch and day: # switch from day to night
                self.switch_allocation()
            if switch and (not day):
                self.reset_allocation()
            
            inci = incident(location, hGrad, np.datetime64(row["tidspunkt"])) 
            inci.lalong = row[["lat", "long"]].to_list()

            if self.toWait:
                
                inci.statusW = row[["t-d", "i-d", "h-l"]].to_list()

        return inci
    def update_ambulances(self, simTime):
        indexes = np.where(self.ambStatus > 0)[0]
        swtime = 0
        if self.surFunc:
            swtime = 1
        for i,a in zip(indexes, np.array(self.ambulances)[indexes]):
            status, available, rTime = a.updateLoc(self.G, simTime)
            self.ambStatus[i] = status
            self.ambLoc[i] = a.gridLoc
            wTime = -1
            if rTime != -1:
                self.responseTime.append(rTime)
                a.responseTime = -1
                wTime = rTime
            elif status == 1:
                wTime = a.incidentAssigned.update_w(simTime)
            if wTime != -1:
                if self.surFunc:
                    wTime = self.dataReader.get_survival(wTime, a.inciPri) # suddenly lower is better.
                    swtime *= wTime
                    #print(wTime, swtime)

                else:
                    wTime = -wTime
                    swtime += wTime
        if self.surFunc:
            inci = [self.dataReader.get_survival(i.update_w(simTime), i.pri) for i in self.incidents]
            if len(inci) > 0:
                swtime *= np.prod(inci)
        return swtime
        #+np.mean(self.responseTime)
        #print(swtime)
        #if len(self.responseTime) > 1:
        #    print(np.mean(self.responseTime))
        #    print(np.quantile(self.responseTime, 0.95))
    def get_random_action(self):
        ambmap = np.zeros(self.size, dtype=int)
        ambBase = np.where(np.isin(self.ambStatus, [0,5]))[0]
        aInds = self.locMap(self.ambLoc[ambBase])
        iInds = [i.index for i in self.incidents]
        return [np.random.choice(aInds), np.random.choice(iInds)]
    def _get_obs(self, tStatus = [0,5]):
        
        ambmap = np.zeros(self.size, dtype=int)
        ambBase = np.where(np.isin(self.ambStatus, tStatus))[0]
        #avail = np.where([not a.sleep for a in self.ambulances])[0]
        #ambBase = 

        self.ambBaseLoc = self.ambLoc[ambBase]

        for i in self.locMap(self.ambBaseLoc):
            ambmap[i] += 1 ### CREATES ERROR when no available ambulances

        indmap = np.zeros(self.size, dtype=int)
        indMapSur = np.zeros(self.size, dtype=float)
        for inci in self.incidents:
            iLoc = inci.gridLoc
            inci.index = self.locMap(iLoc)
            indmap[inci.index] += 1
            if self.surFunc:
                inci.surw = self.dataReader.get_survival(inci.update_w(self.startTime), inci.pri)
                indMapSur[self.locMap(iLoc)] +=  inci.surw # May need to be the minimum.
        if not self.surFunc:
            return {"ambulance": ambmap, "incident": indmap}
        else:
            return {"ambulance": ambmap, "incident": indmap, "incident_sur" : indMapSur}


    def dist_map(self):
        ambBase = np.where(np.isin(self.ambStatus, [0,5]))[0]
        self.ambBaseLoc = self.ambLoc[ambBase]
        
        distmap = np.full(self.size, 1.5, dtype=float)
        amblalo = self.locXY[self.locMap(self.ambBaseLoc), :]
        dists = np.array([np.linalg.norm(np.array(self.incidents[0].lalong) - np.array(l)) for l in amblalo])
        dists = dists/(np.max(dists)+0.1)
        distmap[ambBase] = dists
        return distmap
    def dist_map_haversine(self):
        ambBase = np.where(np.isin(self.ambStatus, [0,5]))[0]
        self.ambBaseLoc = self.ambLoc[ambBase]
        
        distmap = np.full(self.size, 1.5, dtype=float)
        locs = self.locMap(self.ambBaseLoc)
        amblalo = self.locXY[locs, :]
        #dists = np.array([np.linalg.norm(np.array(self.incident.lalong) - np.array(l)) for l in amblalo])
        inciLalo = self.incidents[0].lalong
        #print(self.incident.lalong, amblalo[0])

        dists = np.array([haversine_np(inciLalo[1], inciLalo[0], l[1], l[0]) for l in amblalo])
        dists = dists/(np.max(dists)+0.1)
        distmap[locs] = dists
        return distmap, self.locMap(self.incidents[0].gridLoc)
    
    def dist_map_amb(self):
        ambBase = np.where(np.isin(self.ambStatus, [0,5]))[0]
        self.ambBaseLoc = self.ambLoc[ambBase]
        
        distmap = np.full(self.size, 1.5, dtype=float)
        locs = self.locMap(self.ambBaseLoc)
        amblalo = self.locXY[locs, :]
        #print(self.incident.lalong, amblalo[0])
        dists = np.array([np.linalg.norm(np.array(self.incidents[0].lalong) - np.array(l)) for l in amblalo])
        dists = dists/(np.max(dists)+0.1)
        distmap[locs] = dists
        return distmap, self.locMap(self.incidents[0].gridLoc)
    def _get_info(self):
        if self.surFunc:
            i = np.argmin([inci.surw for inci in self.incidents])
            iInd = self.incidents[i].index

            ambBase = np.where(np.isin(self.ambStatus, [0,5]))[0]
            self.ambBaseLoc = self.ambLoc[ambBase]
            inds = self.locMap(self.ambBaseLoc)
            amblalo = self.locXY[inds, :]
            inciLalo = self.incidents[i].lalong
            dists = np.array([haversine_np(inciLalo[1], inciLalo[0], l[1], l[0]) for l in amblalo])
            aInd = inds[np.argmin(dists)]

            return {"AmbulanceAvail" :  np.where(np.isin(self.ambStatus, [0,5]))[0].shape[0], "AmbulanceAvail2" : np.where(self.ambStatus > 0)[0].shape[0], "recommended" : np.array([aInd, iInd])}
        else:
            return {"AmbulanceAvail" :  np.where(np.isin(self.ambStatus, [0,5]))[0].shape[0], "AmbulanceAvail2" : np.where(self.ambStatus > 0)[0].shape[0]}
        #return {"distMap" : self.dist_map()}
    def update(self):
        self.update_ambulances(self.startTime)
    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, seed=None, options=None):
        np.random.seed(seed=seed)
        #super(gym.Env).reset(seed=seed)
        #self.ambulances = super().prepareAmbulances()
        self.timeSince = np.zeros(len(self.rates[self.rates > 0]))
        self.ambulances = copy.deepcopy(self.ambulances_dist_back)

        if not self.training:
            self.responseTime = []
        else:
            self.responseTime = deque(maxlen=1000)
        self.incidents = [self.gen_incident()]
        #print(self.startTime)
        self.update_ambulances(self.startTime)
        observation = self._get_obs()
        self.currentIter = 0

        #info = self._get_info()

        return observation
    def get_possible(self):
        #return np.where(np.isin(self.ambStatus, [0,5]))[0]
        a = np.where(np.isin(self.ambStatus, [0,5]))[0]
        return a
        #b = np.where([not a.sleep for a in self.ambulances])[0]
        #return a.insersetct1d(a,b)

    def step(self, acList): # action = [ambulanceLoc, IncidentLoc]
        terminal = False
        aInd = np.where(self.locMap(self.ambLoc) == acList[0])[0] # REMEMBER IS AMBLOC ORDER
        possible = self.get_possible()
        aInd = np.intersect1d(possible, aInd)
        

        iInd = np.where(self.locMap([i.gridLoc for i in self.incidents]) == acList[1])[0]

        if aInd.shape[0] > 0 and iInd.shape[0] > 0:
            aInd = np.random.choice(aInd)
            if not self.surFunc:
                iInd = np.random.choice(iInd)
            else:
                min = np.argmin([self.incidents[i].surw for i in iInd])
                iInd = iInd[min]
            #print(aInd, iInd)
        
            incident = self.incidents.pop(iInd)
            self.dispatch(aInd, self.startTime, incident)
            if len(self.incidents) == 0:
                self.incidents.append(self.gen_incident(True))

            wTime = self.update_ambulances(self.startTime)
            possible = self.get_possible()
            times = np.array([a.path.finishedIN for a in self.ambulances if not (a.path is None)])
            #print(times)
            #print([a.sleep for a in self.ambulances])
            #print(self.ambStatus, self.dataReader.day)
            if len(possible) == 0: # wait for ambulances to be available.
                simulating = True
                self.peek = self.get_iTime()
                inciIter = self.peek
                while simulating:
                    times = np.array([a.path.finishedIN for a in self.ambulances if a.status >= 0 and (not (a.path is None))])
                    toMove = np.min(times)*np.timedelta64(60*60, "s")
                    toMove = np.max([toMove, np.timedelta64(1, "s")])
                    #print(toMove)
                    self.startTime = self.startTime+toMove
                    self.peek -= toMove

                    wTime = self.update_ambulances(self.startTime)
                    possible = self.get_possible()
                    simulating = len(possible) == 0
                    print(self.peek, np.timedelta64(0, "s"))
                    while self.peek <= np.timedelta64(0, "s"):
                        print(self.peek)
                        self.incidents.append(self.gen_incident(False, inciIter, peek=self.peek))
                        inciIter = self.get_iTime()
                        self.peek = inciIter+self.peek
           
            if self.fromData:
                terminal = self.dataReader.iter == len(self.dataReader.train.index)-1
            
            if self.training and self.fromData and self.surFunc:
                self.surVals.append(wTime)
            #print(len(self.incidents))
            reward = wTime
            observation = self._get_obs()
            info = self._get_info()
            self.wrongCount = 0

            return observation, reward, terminal, info
        else:
            print("Fuck")
            self.wrongCount += 1
            return self._get_obs(), -10000000*self.wrongCount, True, self._get_info()


        

from typing import Callable
from sb3_contrib.common.wrappers import ActionMasker
def mask_fn(env):
    obs = env._get_obs()
    return np.concatenate([obs["ambulance"] > 0, obs["incident"] > 0], axis=0)

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        #dR = dataReader(training=True)
        env = dispatchEnv(lamb=0.21, LaLo=[59.910986, 10.752496], dist=5, fromData=True, waitTimes=True, surFunc=True)
        #env.dataReader = dR
        env.dataReader.iter = np.random.randint(0,len(env.dataReader.train.index)-500)
        #env.dataReader.iter = len(env.dataReader.train.index)-1000
        env = ActionMasker(env, mask_fn)
        env.seed(seed + rank)
        return env
    return _init       
        
 

def random(env):
    return env.get_random_action()

def min_dist_info(env):
    dMap, inci= env.dist_map_haversine()
    inci = int(inci)
    #print(dMap)
    indices = np.where(dMap == np.min(dMap))[0]
    return [np.random.choice(indices), inci], np.array([[i, inci] for i in indices])

def min_dist_info_euc(env):
    dMap, inci = env.dist_map_amb()
    inci = int(inci)
    indices = np.where(dMap == np.min(dMap))[0]
    return [np.random.choice(indices), inci], np.array([[i, inci] for i in indices])


#a = simulator()
#a.simulate()
#from stable_baselines3.common.env_checker import check_env



        
    
#a.test_2()
#a.test()
# convertion of minutes to hours to fit the graph
# networkx.exception.NodeNotFound: Either source 131.0 or target 4174.0 is not in G

if __name__ == "__main__":
    env = dispatchEnv(lamb=0.21, LaLo=[59.910986, 10.752496], dist=5, fromData=False, waitTimes=True, training=False, surFunc=True)
    for z in range(10000):
        state = env._get_obs()
        print(np.where(state["ambulance"] > 0)[0])
        a = np.random.choice(np.where(state["ambulance"] > 0)[0])
        i = np.random.choice(np.where(state["incident"] > 0)[0])
        env.step([a,i])
