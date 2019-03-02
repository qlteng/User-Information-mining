# coding: utf-8

import configparser
import codecs
def utf8_encode(params):
    pass
def config_parse(conf_path):

    conf = configparser.ConfigParser()
    conf.read(conf_path)

    # conf
    datasource = conf.get("config", "datasource")
    types = conf.get("config", "types")
    n_steps = conf.get("config", "n_steps")
    n_channel = conf.get("config", "n_channel")
    n_class = conf.get("config", "n_class")
    overlap = conf.get("config", "overlap")
    target = conf.get("config", "target")
    process_num = conf.get("config", "process_num")

    # filter
    phonetype = conf.get("filter", "phonetype")
    phoneposition = conf.get("filter", "phoneposition")
    activity = conf.get("filter", "activity")

    params = [datasource, types, n_steps, n_channel, n_class, overlap, target, process_num, phonetype, phoneposition, activity]
    datasource, types, n_steps, n_channel, n_class, overlap, target, process_num, phonetype, phoneposition, activity = map(lambda x: x.encode('utf-8'), params)
    filter = {"phonetype" : phonetype, "phoneposition" : phoneposition, "activity" : activity}
    return datasource, types, n_steps, n_channel, n_class, overlap, target, process_num, filter

if __name__ == '__main__':

    path = "../model.conf"
    datasource, types, n_steps, n_channel, n_class, overlap, target, process_num, filter = config_parse(path)
    print overlap

