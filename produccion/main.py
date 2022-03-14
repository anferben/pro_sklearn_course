from utils import Utils
from models import Models


if __name__ == '__main__':
    utils = Utils()
    models = Models()

    data = utils.load_from_csv('./in/felicidad.csv')
    
    drop_cols = ['country', 'rank', 'score']
    y_col = ['score']
    X, y = utils.features_target(data, drop_cols=drop_cols, y_col=y_col)

    models.grid_training(X, y)