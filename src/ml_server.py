from flask import Flask, url_for
from flask_wtf import FlaskForm
from flask import render_template, redirect, send_file
from flask_bootstrap import Bootstrap
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, SelectField, FileField

import pandas as pd

from ensembles import RandomForestMSE, GradientBoostingMSE
from utils import DataPreprocessor


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)

model_choices = [('rf', 'Random Forest'),
                 ('gb', 'Gradient Boosting')]


class ModelForm(FlaskForm):
    model_class = SelectField('Model type', choices=model_choices,
                              validators=[DataRequired()])
    submit = SubmitField('Next')


class RFParamsForm(FlaskForm):
    n_estimators = StringField('n_estimators', validators=[DataRequired()])
    max_depth = StringField('max_depth', validators=[DataRequired()])
    feature_subsample_size = StringField('feature_subsample_size', validators=[DataRequired()])
    submit = SubmitField('Create model')


class GBParamsForm(FlaskForm):
    n_estimators = StringField('n_estimators', validators=[DataRequired()])
    learning_rate = StringField('learning_rate', validators=[DataRequired()])
    max_depth = StringField('max_depth', validators=[DataRequired()])
    feature_subsample_size = StringField('feature_subsample_size', validators=[DataRequired()])
    submit = SubmitField('Create model')


class TrainingFileForm(FlaskForm):
    file = FileField('Training file', validators=[DataRequired()])
    submit = SubmitField('Train!')


class FeaturesForm(FlaskForm):
    num_features = StringField('Numerical features')
    bin_features = StringField('Binary features')
    cat_features = StringField('Categorical features')


class TrainedModelMenuForm(FlaskForm):
    val_file = FileField('Validation file', validators=[DataRequired()])
    evaluate = SubmitField('Evaluate!')
    test_file = FileField('Test file', validators=[DataRequired()])
    predict = SubmitField('Predict!')


model = None
model_class = None
data_transformer = None


@app.route('/', methods=['GET', 'POST'])
def choose_model():
    model_form = ModelForm()

    if model_form.validate_on_submit():
        global model_class
        model_class_name = model_form.model_class.data

        if model_class_name == 'rf':
            model_class = RandomForestMSE
        elif model_class_name == 'gb':
            model_class = GradientBoostingMSE
        else:
            raise ValueError('Invalid model class')

        return redirect(url_for('choose_params'))
    return render_template('from_form.html', form=model_form)


@app.route('/params', methods=['GET', 'POST'])
def choose_params():
    if model_class == RandomForestMSE:
        param_form = RFParamsForm()
    else:
        param_form = GBParamsForm()

    if param_form.validate_on_submit():
        model_params = {'n_estimators': int(param_form.n_estimators.data)}
        if model_class == GradientBoostingMSE:
            model_params['learning_rate'] = float(param_form.learning_rate.data)
        model_params['max_depth'] = int(param_form.max_depth.data)
        model_params['feature_subsample_size'] = int(param_form.feature_subsample_size.data)
        global model
        model = model_class(**model_params)
        return redirect(url_for('training'))
    return render_template('from_form.html', form=param_form)


@app.route('/training', methods=['GET', 'POST'])
def training():
    training_file_form = TrainingFileForm()
    features_form = FeaturesForm()
    if features_form.validate_on_submit():
        train_data = pd.read_csv(training_file_form.file.data)

        num_features = features_form.num_features.data.split(', ') if features_form.num_features.data else []
        bin_features = features_form.bin_features.data.split(', ') if features_form.bin_features.data else []
        cat_features = features_form.cat_features.data.split(', ') if features_form.cat_features.data else []
        print(num_features, bin_features, cat_features)
        if not num_features and not bin_features and not cat_features:
            data_transformer = DataPreprocessor(mode='auto')
        else:
            data_transformer = DataPreprocessor('manual', num_features, bin_features, cat_features)

        y_train = train_data['target'].to_numpy()

        train_data = train_data.drop(columns=['target'])
        X_train = data_transformer.fit_transform(train_data)

        model.fit(X_train, y_train)

        return redirect(url_for('trained_model_menu'))
    return render_template('from_training_form.html',
                           training_file_form=training_file_form,
                           features_form=features_form)


@app.route('/trained_model_menu', methods=['GET', 'POST'])
def trained_model_menu():
    menu_form = TrainedModelMenuForm()
    if menu_form.validate_on_submit():
        if menu_form.predict.data:
            X_test = pd.read_csv(menu_form.test_file.data)
            y_pred = model.predict(X_test)
            y_pred.to_csv('prediction.csv')
            return send_file('prediction.csv', as_attachment=True)
        elif menu_form.evaluate.data:
            val_data = pd.read_csv(menu_form.val_file.data)
            y_val = val_data['target']
            X_val = val_data.drop(columns=['target'])
            y_pred = model.predict(X_val)
            redirect(url_for('evaluation'))
    return render_template('from_form.html', form=menu_form)


@app.route('/evaluation', methods=['GET', 'POST'])
def evaluation():
    training_file_form = TrainingFileForm()
    if training_file_form.validate_on_submit():
        training_data = pd.read_csv(training_file_form.file.data)
        y_train = training_data['target']
        X_train = training_data.drop(columns=['target'])
        model.fit(X_train, y_train)
        return redirect(url_for('trained_model_menu'))
    return render_template('from_form.html', form=training_file_form)
