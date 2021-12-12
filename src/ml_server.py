from flask import Flask, url_for, render_template, redirect, send_file, flash
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, SelectField, FileField

import pandas as pd
import numpy as np
import plotly
import plotly.subplots
import plotly.graph_objects as go

from ensembles import RandomForestMSE, GradientBoostingMSE
from utils import DataPreprocessor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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
    val_fraction = StringField('Validation fraction')
    val_file = FileField('Validation file')
    submit = SubmitField('Train!')


class TestForm(FlaskForm):
    test_file = FileField('Test file')
    predict = SubmitField('Predict!')


model = None
data_transformer = None
train_dataset_name = None
val_dataset_name = None
val_fraction = None
hist = None


@app.route('/', methods=['GET', 'POST'])
def model_creation_page():
    model_form = ModelForm()
    error = None
    if model_form.validate_on_submit():
        try:
            n_estimators = int(model_form.n_estimators.data)
            max_depth = int(model_form.max_depth.data)
            feature_subsample_size = float(model_form.feature_subsample_size.data)
            if n_estimators <= 0 or max_depth <= 0:
                raise ValueError('Parameters must be positive')
            if feature_subsample_size <= 0 or feature_subsample_size > 1:
                raise ValueError('Feature subsample size must be from 0 to 1')

            model_params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                            'feature_subsample_size': feature_subsample_size}

            model_type = model_form.model_type.data
            global model
            if model_type == 'rf':
                model = RandomForestMSE(**model_params)
            elif model_type == 'gb':
                if not model_form.learning_rate.data:
                    raise ValueError('No learning rate for Gradient Boosting')
                learning_rate = float(model_form.learning_rate.data)
                if learning_rate <= 0:
                    ValueError('Learning rate must be positive')
                model_params['learning_rate'] = learning_rate
                model = GradientBoostingMSE(**model_params)
            else:
                raise ValueError('Invalid model type')
            return redirect(url_for('train_page'))
        except Exception as e:
            error = e
    return render_template('model_creation_page.html', model_form=model_form, error=(repr(error) if error else None))


@app.route('/model', methods=['GET', 'POST'])
def train_page():
    if not model:
        return render_template('no_model_page.html')
    train_val_form = TrainValForm()
    error = None
    if train_val_form.validate_on_submit():
        try:
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

            global hist
            global val_dataset_name
            global val_fraction
            if train_val_form.val_file.data:
                val_dataset_name = train_val_form.val_file.data.filename
                val_fraction = 1
                val_data = pd.read_csv(train_val_form.val_file.data)
                y_val = val_data['target'].to_numpy()
                val_data = val_data.drop(columns=['target'])
                X_val = data_transformer.transform(val_data)
                hist = model.fit(X_train, y_train, X_val, y_val)
            elif train_val_form.val_fraction.data:
                val_dataset_name = train_dataset_name
                val_fraction = float(train_val_form.val_fraction.data)
                if val_fraction < 0 or val_fraction >= 1:
                    raise ValueError('Validation fraction must be between 0 and 1')
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_fraction,
                                                                  random_state=0)
                hist = model.fit(X_train, y_train, X_val, y_val)
            else:
                val_dataset_name = 'No validation dataset'
                val_fraction = None
                hist = model.fit(X_train, y_train)

            return redirect(url_for('main_page'))
        except Exception as e:
            error = e
    params = {'model_class': type(model).__name__}
    params.update(model.get_params())
    return render_template('train_page.html', params_dict=params,
                           train_val_form=train_val_form, error=(repr(error) if error else None))


@app.route('/trained_model', methods=['GET', 'POST'])
def main_page():
    if not model:
        return render_template('no_model_page.html')
    test_form = TestForm()
    error = None
    if test_form.validate_on_submit():
        try:

            test_data = pd.read_csv(test_form.test_file.data)
            X_test = data_transformer.transform(test_data)
            y_pred = model.predict(X_test)
            np.savetxt('prediction.txt', y_pred, delimiter='\n')

            return send_file('prediction.txt', as_attachment=True)
        except Exception as e:
            error = e
    params = {'model_class': type(model).__name__}
    params.update(model.get_params())
    return render_template('main_page.html', params_dict=params,
                           test_form=test_form, train_dataset_name=train_dataset_name,
                           val_dataset_name=val_dataset_name, val_fraction=str(val_fraction),
                           error=(repr(error) if error else None))


@app.route('/evaluation', methods=['GET', 'POST'])
def eval_page():
    if not model:
        return render_template('no_model_page.html')
    if not hist:
        return render_template('ho_hist_page.html')
    # hist = {'train_rmse': [1, 2, 3], 'val_rmse': [2, 3, 4], 'time': [1, 2, 3], 'n_estimators': [1, 2, 3]}

    fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                        subplot_titles=['Train and val RMSE for each iteration',
                                                        'Total time for each iteration'],
                                        vertical_spacing=0.1)

    fig.add_trace(go.Scatter(
        x=hist['n_estimators'],
        y=hist['train_rmse'],
        name='Train'
    ), row=1, col=1)
    if 'val_rmse' in hist:
        fig.add_trace(go.Scatter(
            x=hist['n_estimators'],
            y=hist['val_rmse'],
            name='Validation'
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist['n_estimators'],
        y=hist['time'],
        name='Time'
    ), row=2, col=1)

    fig.update_xaxes(
        title_text='Iteration',
        row=1, col=1)

    fig.update_yaxes(
        title_text='RMSE',
        row=1, col=1)

    fig.update_xaxes(
        title_text='Iteration',
        row=2, col=1)

    fig.update_yaxes(
        title_text='Time (s)',
        row=2, col=1)

    fig.update_layout(
        width=1200, height=1200
    )

    return render_template('eval_page.html', graph_div=fig.to_html(full_html=False),
                           train_rmse=f"{hist['train_rmse'][-1]:.2f}",
                           test_rmse=f"{hist['val_rmse'][-1]:.2f}" if 'val_rmse' in hist else 'None',
                           time=f"{hist['time'][-1]:.2f}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
