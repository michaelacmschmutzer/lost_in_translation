#!/usr/bin/env python3

"""Make the SQLite database for cluster simulations"""

import os
import numpy as np
import sqlite3 as sql
import evolution as evo

def make_mistranslation_database(directory, dbname="gb1"):
    flist = os.listdir(directory)
    db = sql.connect("../data/"+dbname+".db")
    cur = db.cursor()
    cur.execute("CREATE TABLE {} (seq, fmean, fvar, psum)".format(dbname))
    try:
        for f in flist:
            values = np.loadtxt(directory+f)
            seq = f.split(".")[0]
            fmean = values[0]
            fvar  = values[1]
            psums = values[2]
            cur.execute('''INSERT INTO {} VALUES (?,?,?,?)'''.format(dbname),
                            (seq, fmean, fvar, psums) )
        # Creating an index greatly speeds up retrieval using nt sequences
        cur.execute("CREATE INDEX Idx1 ON {}(seq)".format(dbname))
        db.commit()
    except Exception as e:
        print(e)
    finally:
        db.close()

def make_no_mistranslation_database(gb1, directory, dbname="gb1nomis"):
    db = sql.connect("../data/"+dbname+".db")
    cur = db.cursor()
    cur.execute("CREATE TABLE {} (seq, aaseq, fitness)".format(dbname))
    for aaseq, fit in gb1.items():
        ntseqs = evo.reverse_translate(aaseq)
        for nt in ntseqs:
            cur.execute('''INSERT INTO {} VALUES (?,?,?)'''.format(dbname),
                            (nt, aaseq, fit))
    cur.execute("CREATE INDEX Idx1 on {}(seq)".format(dbname))
    db.commit()
    db.close()
