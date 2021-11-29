from flask import Flask, url_for
from flask_wtf import FlaskForm
from flask import render_template, redirect
from flask_bootstrap import Bootstrap
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, SelectField

from ensembles import RandomForestMSE, GradientBoostingMSE


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


model = None
model_class = None


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
        model_params = {}
        model_params['n_estimators'] = int(param_form.n_estimators.data)
        if model_class == GradientBoostingMSE:
            model_params['learning_rate'] = float(param_form.learning_rate.data)
        model_params['max_depth'] = int(param_form.max_depth.data)
        model_params['feature_subsample_size'] = int(param_form.feature_subsample_size.data)
        global model
        model = model_class(**model_params)
        # return redirect(url_for('training'))
    return render_template('from_form.html', form=param_form)
