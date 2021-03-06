# coding: utf-8

import os
import multiprocessing
import json
import datetime
import mysqlDAL
import logging

LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

def query(process_metaId, index, store_path):

    label = {}
    data = {}
    DAL = mysqlDAL.mysqlDAL(host ='localhost', user ='other', passwd ='pkl239', db ='qltenghasc')
    iter = 0

    datafile = "%s/dump_data%d.json" % (store_path, index)
    labelfile = "%s/dump_label%d.json" % (store_path, index)

    start_time = datetime.datetime.now()
    for sid, uid, gender, generation, height, weight, position, phone, mount, aid in process_metaId:

        label[sid] = '#'.join([str(uid), str(gender), str(generation), str(height), str(weight), position, phone, mount, aid])

        sql = 'select acc.X,acc.Y,acc.Z,gyro.X,gyro.Y,gyro.Z  \
              from acc,gyro  \
              where acc.MetaId=%d  \
              and acc.MetaId=gyro.MetaId  \
              and acc.SamplingTime=gyro.SamplingTime' % sid

        DAL.execute(sql, metaid = sid)
        data[sid] = list(DAL.cursor.fetchall())
        iter += 1
        if iter % 20 == 0:
            print "Process index %d dump %d samples"%(index, iter)

    with open(datafile, 'w') as f_data:
        json.dump(data, f_data)
    with open(labelfile, 'w') as f_label:
        json.dump(label, f_label)

    timecost = datetime.datetime.now() - start_time
    print "Query time cost :%s"%timecost

    DAL.close()


def dump_from_mysql(store_path):

    DAL = mysqlDAL.mysqlDAL(host ='localhost', user ='other', passwd ='pkl239', db ='qltenghasc')

    #   for activity and attribute
    #   no person in

    # for authentication:
    #     person in 14 userlist on all activity
    #     person in 95 userlist on walk whose has more than 10 samples

    # sql = 'select Id,Person,Gender,Generation,Height,Weight,TerminalPosition,TerminalType,TerminalMount,Activity \
    #         from meta \
    #         where HasAcc=1 \
    #         and HasGyro=1 \
    #         and TerminalPosition in \
    #         ("arm;hand","arm;right;hand","bag","bag;position(fixed);backpack","bag;position(fixed);handback",\
    #         "bag;position(fixed);messengerbag","bag;position(fixed);shoulderbag","strap;waist;rear","waist",\
    #         "wear;outer;chest","wear;outer;chest;left","wear;outer;waist;front","wear;pants;front",\
    #         "wear;pants;waist;fit;left-front","wear;pants;waist;fit;right-back","wear;pants;waist;fit;right-front") \
    #         and TerminalType in \
    #         ("Logger+Wifi for Android;1.0","SAMSUNG; Galaxy Nexus; Android 4.1.2","Samsung;Galaxy Nexus;Android OS 4.1.2",\
    #         "Samsung;Galaxy Nexus;AndroidOS 4.1;","Samsung;Nexus S;Android OS 4.1.2","Samsung;NexusS;AndroidOS 4.1;")\
    #         and Activity in ("walk") \
    #         and Person in \
    #         (select Person from meta where HasAcc=1 and HasGyro=1 and Activity in ("walk") and TerminalPosition in \
    #         ("arm;hand","arm;right;hand","bag","bag;position(fixed);backpack","bag;position(fixed);handback",\
    #             "bag;position(fixed);messengerbag","bag;position(fixed);shoulderbag","strap;waist;rear","waist",\
    #             "wear;outer;chest","wear;outer;chest;left","wear;outer;waist;front","wear;pants;front",\
    #             "wear;pants;waist;fit;left-front","wear;pants;waist;fit;right-back","wear;pants;waist;fit;right-front") \
    #             and TerminalType in \
    #             ("Logger+Wifi for Android;1.0","SAMSUNG; Galaxy Nexus; Android 4.1.2","Samsung;Galaxy Nexus;Android OS 4.1.2",\
    #             "Samsung;Galaxy Nexus;AndroidOS 4.1;","Samsung;Nexus S;Android OS 4.1.2","Samsung;NexusS;AndroidOS 4.1;") \
    #             group by Person having count(*)>=10);'
    sql = 'select Id,Person,Gender,Generation,Height,Weight,TerminalPosition,TerminalType,TerminalMount,Activity \
            from meta \
            where HasAcc=1 \
            and HasGyro=1 \
            and TerminalPosition like "%waist%" \
            and Activity in ("walk") \
            and Person in \
            (select Person from meta where HasAcc=1 and HasGyro=1 and Activity in ("walk") and TerminalPosition like "%waist%" group by Person having count(*)>15)'

    while True:

        DAL.execute(sql)
        result = DAL.cursor.fetchall()
        total_data_num = len(result)
        process_num = 40
        print "Total number is : %d" % total_data_num
        tasks_per_process = (total_data_num // process_num) + 1
        record = []
        for index in xrange(0, process_num):
            begin = index * tasks_per_process
            end = (index + 1) * tasks_per_process
            process_metaId = result[begin : end]

            p = multiprocessing.Process(target = query, args = (process_metaId, index, store_path))
            p.start()
            record.append(p)
        for process in record:
            process.join()

        DAL.close()
        break
if __name__ == '__main__':
    store_path = "../../data/hasc/human_walk_on_waist_14_data"
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    dump_from_mysql(store_path)