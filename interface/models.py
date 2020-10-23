from interface import db
from datetime import datetime

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task = db.Column(db.String)
    worker_id = db.Column(db.String, default='')
    pages = db.relationship('Page', backref='user', lazy='dynamic')
    time_created = db.Column(db.DateTime, default=datetime.utcnow)


class Page(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    page_num = db.Column(db.Integer)
    keyword = db.Column(db.String)
    nn1 = db.Column(db.String)
    nn2 = db.Column(db.String)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, onupdate=datetime.utcnow)
    pos1 = db.Column(db.String, default='')
    pos2 = db.Column(db.String, default='')
    neg1 = db.Column(db.String, default='')
    neg2 = db.Column(db.String, default='')
    new1 = db.Column(db.String, default='')
    new2 = db.Column(db.String, default='')





