from obspy import read_inventory

inv = read_inventory("/DATA/VOILA/metadata/*/*.xml")

w = open("stations_voila.txt", "w")

for net in inv:
    for sta in net:
        w.write("{} {} {} {}\n".format(net.code, sta.code, sta.latitude, sta.longitude))
