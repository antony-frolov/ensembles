from flask import Flask, url_for
from flask_wtf import FlaskForm
from flask import render_template, redirect, send_file
from flask_bootstrap import Bootstrap
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, SelectField, FileField

import pandas as pd
import numpy as np

from ensembles import RandomForestMSE, GradientBoostingMSE
from utils import DataPreprocessor

from sklearn.metrics import mean_squared_error


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)

model_choices = [('rf', 'Random Forest'),
                 ('gb', 'Gradient Boosting')]


class ModelForm(FlaskForm):
    model_type = SelectField('Model type', choices=model_choices,
                             validators=[DataRequired()])
    n_estimators = StringField('Number of estimators', validators=[DataRequired()])
    learning_rate = StringField('Learning rate (leave blank for Random Forest)')
    max_depth = StringField('Maximum tree depth', validators=[DataRequired()])
    feature_subsample_size = StringField('Feature subsample size', validators=[DataRequired()])
    submit = SubmitField('Create model!')


class TrainValForm(FlaskForm):
    num_features = StringField('Numerical features')
    bin_features = StringField('Binary features')
    cat_features = StringField('Categorical features')
    train_file = FileField('Train file', validators=[DataRequired()])
    val_file = FileField('Validation file')
    submit = SubmitField('Train!')


class TestForm(FlaskForm):
    test_file = FileField('Test file')
    predict = SubmitField('Predict!')


model = None
data_transformer = None
train_dataset_name = None
scores = None


@app.route('/', methods=['GET', 'POST'])
def model_creation_page():
    model_form = ModelForm()

    if model_form.validate_on_submit():
        model_params = {'n_estimators': int(model_form.n_estimators.data),
                        'max_depth': int(model_form.max_depth.data),
                        'feature_subsample_size': int(model_form.feature_subsample_size.data)}
        model_type = model_form.model_type.data
        global model
        if model_type == 'rf':
            model = RandomForestMSE(**model_params)
        elif model_type == 'gb':
            if not model_type.learning_rate.data:
                raise ValueError()
            model_params['learning_rate'] = float(model_type.learning_rate.data)
            model = GradientBoostingMSE(**model_params)
        else:
            raise ValueError('Invalid model type')

        return redirect(url_for('train_page'))
    return render_template('model_creation_page.html', model_form=model_form)


@app.route('/model', methods=['GET', 'POST'])
def train_page():
    train_val_form = TrainValForm()
    if train_val_form.validate_on_submit():
        global train_dataset_name
        train_dataset_name = train_val_form.train_file.data.filename
        train_data = pd.read_csv(train_val_form.train_file.data)

        num_features = train_val_form.num_features.data.split(', ') if train_val_form.num_features.data else []
        bin_features = train_val_form.bin_features.data.split(', ') if train_val_form.bin_features.data else []
        cat_features = train_val_form.cat_features.data.split(', ') if train_val_form.cat_features.data else []

        global data_transformer
        if not num_features and not bin_features and not cat_features:
            data_transformer = DataPreprocessor(mode='auto')
        else:
            data_transformer = DataPreprocessor('manual', num_features, bin_features, cat_features)

        y_train = train_data['target'].to_numpy()

        train_data = train_data.drop(columns=['target'])
        X_train = data_transformer.fit_transform(train_data)

        global scores
        if train_val_form.val_file.data:
            val_data = pd.read_csv(train_val_form.val_file.data)
            y_val = val_data['target'].to_numpy()
            val_data = val_data.drop(columns=['target'])
            X_val = data_transformer.transform(val_data)
            scores = model.fit(X_train, y_train, X_val, y_val)
        else:
            scores = model.fit(X_train, y_train)

        return redirect(url_for('main_page'))
    return render_template('train_page.html', params_dict=model.get_params(), train_val_form=train_val_form)


@app.route('/trained_model', methods=['GET', 'POST'])
def main_page():
    test_form = TestForm()
    if test_form.validate_on_submit():
        test_data = pd.read_csv(test_form.test_file.data)
        X_test = data_transformer.transform(test_data)
        y_pred = model.predict(X_test)
        np.savetxt('prediction.txt', y_pred, delimiter='\n')
        return send_file('prediction.txt', as_attachment=True)
    return render_template('main_page.html', params_dict=model.get_params(),
                           test_form=test_form, train_dataset_name=train_dataset_name)


@app.route('/evaluation', methods=['GET', 'POST'])
def eval_page():
    return render_template('eval_page.html', scores=scores)
