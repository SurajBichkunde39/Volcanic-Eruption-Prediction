import os
import pickle


from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import pandas as pd


from . import utils
from .model import Model

static_url_path = os.path.join(os.getcwd(), 'app/static')
app = Flask(__name__, static_url_path=static_url_path)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]iasdfffsd/'

ALLOWED_EXTENSIONS = set(["csv"])
utils.get_files()
model = Model()


def allowed_file(filename):
    mask1 = "." in filename
    mask2 = filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    return mask1 and mask2


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part", 'error')
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            flash("No selected file", 'warning')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            UPLOAD_FOLDER = os.path.join(os.getcwd(), "app/upload_dir/")
            # final_filename = os.path.join(UPLOAD_FOLDER, filename)
            filename = filename.split('.')
            middle_str = utils.generate_ranom_string()
            new_filename = filename[0]+middle_str+'.'+filename[1]
            final_filename = os.path.join(UPLOAD_FOLDER, new_filename)
            file.save(final_filename)
            df = pd.read_csv(final_filename)
            result_dict = utils.check_dataframe(df)
            if result_dict['all_ok'] == "okay":
                # everything is good
                features = utils.preprocess_dataframe(df)
                # construct middle path i.e. name of the files
                mid_path_img = middle_str + '.png'
                mid_path_stat = middle_str + '.csv'
                # Get full paths save the files
                fig_path = os.path.join(os.getcwd(), 'app/static/plots')
                fig_path = os.path.join(fig_path, mid_path_img)
                stats_path = os.path.join(os.getcwd(), 'app/stats')
                stats_path = os.path.join(stats_path, mid_path_stat)
                # call predict and plot
                # they will internally create the files
                time_to_eruption = model.predict(features, stats_path)
                utils.plot_the_sensors(df, fig_path)
                return redirect(url_for('show_results', key=middle_str,
                                        time_to_eruption=time_to_eruption))
            else:
                errors = []
                for a, b in result_dict.items():
                    if a != 'all_ok':
                        errors.append(':'.join([a, b]))
                for error in errors:
                    flash(errors, 'error')
                return redirect(request.url)
            return redirect(url_for("uploaded_file", filename=filename))
        else:
            flash("Either file is not a csv or the fortmat is not right",'error')
            flash("Please try again", 'info')
            return redirect(request.url)
    return render_template('index.html')


@app.route("/results", methods=["GET", "POST"])
def show_results():
    key = request.args.get('key')
    time_to_eruption = request.args.get('time_to_eruption')
    path_to_image = 'plots/' + key + '.png'
    path_to_stats = os.path.join(os.getcwd(), f'app/stats/{key}.csv')
    stats = pd.read_csv(path_to_stats).iloc[0]
    imp_feat = pickle.load(open(model.feat_imp_path, 'rb'))
    features_to_display = {}
    for a, b in imp_feat.items():
        features_to_display[a] = [b]
        features_to_display[a].append(stats[a])
    content = {
        'path_to_image': path_to_image,
        'features_to_display': features_to_display,
        'time_to_eruption': time_to_eruption}
    return render_template("result.html", content=content)


@app.route('/sample_file')
def send_sample_file():
    rel_path_sample = 'app/sample_data/1000213997.csv'
    path_to_sample_data = os.path.join(os.getcwd(), rel_path_sample)
    return send_file(path_to_sample_data, mimetype='text/csv',
                     attachment_filename='Sample_data.csv')
