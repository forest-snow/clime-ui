import flask
from interface import app
from interface import load, save
from interface import app, db
from interface.models import User, Page
import json
import pickle

task_language_labels = {
    'sentiment':{'src':'ENGLISH','tgt':'CHINESE'},
    'rcv_zh':{'src':'ENGLISH','tgt':'CHINESE'},
    'rcv_ru':{'src':'ENGLISH','tgt':'RUSSIAN'},
}


@app.route('/<string:task>/<string:worker_id>')
def start(task, worker_id):
    """ Start page where user entry is created/reloaded from database """

    # see if session is already created
    try:
        uid = flask.session['uid']
        user = User.query.get(uid)

        if user is None:
            user = User(task=task, worker_id=worker_id)
            db.session.add(user)
            db.session.commit()

    except KeyError:
        # create user
        user = User(task=task, worker_id=worker_id)
        db.session.add(user)
        db.session.commit()

    flask.session['uid'] = user.id

    if user.worker_id != worker_id:
        user.worker_id = worker_id
        db.session.commit()

    # add page information if not in database
    if len(user.pages.all()) == 0:    
        load.setup_to_page(flask.session['uid'], task)

    return flask.render_template('start.html', user=user.id)


@app.route('/ui/<int:entry>')
def ui(entry):
    uid = flask.session['uid']
    user = User.query.get(uid)
    row = load.load_page(user, entry)
    total = len(user.pages.all())
    page = {'current':entry+1, 'total':total}
    lang_labels = task_language_labels[user.task]
    return flask.render_template('ui2.html', row=row, user=uid, page=page, lang_labels=lang_labels)


@app.route('/save/<int:entry>', methods=['POST'])
def save_entry(entry):
    """ Save user feedback about word with ID [entry] in database"""
    data = flask.request.get_json()
    print(data)
    uid = flask.session['uid']
    user = User.query.get(uid)
    # find latest update from user on interface
    save.save_results(data, user ,entry)

    return "success"


@app.route('/autocomplete/<int:language>')
def autocomplete(language):
    uid = flask.session['uid']
    user = User.query.get(uid)

    limit = 8
    query = flask.request.args.get('query')
    vocab = load.load_vocab(language, user.task)
    choices = []
    for w in vocab:
        if len(choices) >= limit:
            break
        if w.startswith(query):
            choices.append(w)
    print(choices)
    return flask.jsonify(choices=choices)

@app.route('/context/<int:language>')
def context(language):
    """ Get concordance from JSON file """
    uid = flask.session['uid']
    user = User.query.get(uid)

    query = flask.request.args.get('query')
    doc = load.concordance(query, language, user.task)
    return flask.jsonify(doc=doc)

@app.route('/restart')
def restart():
    """ Restart interface by deleting all previous feedback from user """
    user = User.query.get(flask.session['uid'])
    task = user.task
    worker_id = user.worker_id
    for page in user.pages.all():
        db.session.delete(page)
    db.session.commit()
    return flask.redirect('/{}/{}'.format(task, worker_id))

@app.route('/finish')
def finish():
    """ Clear flask session for new session """
    uid = flask.session['uid']
    flask.session.clear()
    return flask.render_template('finish.html', user=uid)

def debug_update(update):
    print('pos1: {}'.format(update.pos1))
    print('pos2: {}'.format(update.pos2))
    print('neg1: {}'.format(update.neg1))
    print('neg2: {}'.format(update.neg2))


@app.route('/instruction')
def instruction():
    return flask.render_template('instruction.html')
