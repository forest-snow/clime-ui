import utils
import json
from interface.models import User, Page
from interface import db

def append_to_results(query, prev_json, new_data):
    prev = json.loads(prev_json)
    new = {query:[word for word in new_data]}
    results = {**prev, **new}
    return json.dumps(results)

def save_results(data, user, entry, debug=True):
    page = user.pages.filter_by(page_num=entry).first()
    page.pos1 = json.dumps(data['pos1'])
    page.neg1 = json.dumps(data['neg1'])
    page.new1 = json.dumps(data['new1'])
    page.pos2 = json.dumps(data['pos2'])
    page.neg2 = json.dumps(data['neg2'])
    page.new2 = json.dumps(data['new2'])
    db.session.commit()

    if debug:
        new_page = user.pages.filter_by(page_num=entry).first()
        print('pos1: {}'.format(new_page.pos1))
        print('neg1: {}'.format(new_page.neg1))
        print('new1: {}'.format(new_page.new1))
        print('pos2: {}'.format(new_page.pos2))
        print('neg2: {}'.format(new_page.neg2))
        print('new2: {}'.format(new_page.new2))

