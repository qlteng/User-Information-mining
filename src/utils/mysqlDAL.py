# coding: utf-8


import MySQLdb
import logging
LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class mysqlDAL:

    def __init__(self, host, user, passwd, db):

        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db

        self.conn = None
        self.cursor = None

    def open(self):

        self.conn = MySQLdb.connect(self.host, self.user, self.passwd, self.db)
        self.cursor = self.conn.cursor()

    def get_cursor(self):
        if not self.cursor:
            self.open()
        return self.cursor

    def execute(self, sql, *params, **kwargs):
        retries = kwargs.get('retries', 1)
        MetaId = kwargs.get('metaid', 0)
        while True:
            try:
                cursor = self.get_cursor()
                cursor.execute(sql, params or None)
                return cursor
            except:
                logging.warning("Mysql Execute Error!")
                if MetaId != 0:
                    logging.warning("MetaId:%d Fetch Failed" % MetaId)
                self.close
                if retries <= 0:
                    raise
                retries -= 1

    def close(self):

        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.conn:
            self.conn.close()
            self.conn = None