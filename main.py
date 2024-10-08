from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

app = Flask(__name__)

# Charger et prétraiter les données lorsque l'application démarre
def load_data():
    df = pd.read_csv('Donn_es_Transform_es_F1.csv')

    # Convertir les chaînes de temps en secondes
    def convert_to_seconds(time_str):
        try:
            if isinstance(time_str, str) and ':' in time_str:
                minutes, rest = time_str.split(':')
                seconds, milliseconds = rest.split('.')
                total_seconds = int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
                return total_seconds
            else:
                return time_str
        except ValueError:
            return np.nan

    df['avg_time_lap'] = df['avg_time_lap'].apply(convert_to_seconds)
    df['fastestLapTime'] = df['fastestLapTime'].apply(convert_to_seconds)

    df.replace('\\N', np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    X = df[['circuitId', 'avg_time_lap', 'fastestLapTime',
            'rainy', 'high_temperature', 'strong_wind',
            'TempAir', 'TempTrack', 'Humidity',
            'WindSpeed', 'WindDirection', 'driverId', 'startingPosition']]
    y = df[['position']]

    X = X.apply(pd.to_numeric, errors='coerce')

    # Diviser et normaliser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Entraîner le modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train.values.ravel())

    # Test sur les données de test (20 % des données)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # Métriques pour l'ensemble de test
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    ndcg_val = ndcg_score(np.asarray([y_test.values.flatten()]), np.asarray([y_pred.flatten()]), k=10)

    print(f"Test MAE: {mae}")
    print(f"Test MSE: {mse}")
    print(f"NDCG@10: {ndcg_val}")

    # Validation croisée (5-fold)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(model, X_train_scaled, y_train.values.ravel(), cv=kfold, scoring='neg_mean_absolute_error')

    print(f"Validation croisée MAE: {-cross_val_scores.mean()} (écart-type: {cross_val_scores.std()})")

    # Stocker dans un dictionnaire
    data = {
        'df': df,
        'scaler': scaler,
        'model': model,
        'X_train_columns': X_train.columns,
        'X_test_scaled': scaler.transform(X_test),
        'y_test': y_test
    }

    return data

data = load_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        circuit_name = request.form.get('circuit_name')
        temp = float(request.form.get('temp'))
        temp_track = float(request.form.get('temp_track'))
        humidity = float(request.form.get('humidity'))
        wind_speed = float(request.form.get('wind_speed'))
        rain = request.form.get('rain')

        # Récupérer les positions de départ via les champs cachés
        starting_positions = {}
        for i in range(1, 23):
            pilot = request.form.get(f'position_{i}')
            if pilot:
                starting_positions[i] = pilot

        print("Positions Reçues:", starting_positions)  # Ajout du log

        # Vérifier que 22 pilotes ont été sélectionnés
        if len(starting_positions) != 22:
            error = f"Vous devez sélectionner exactement 22 pilotes."
            return render_template('index.html', error=error, circuits=data['df']['circuitName'].unique(), pilotes=data['df']['driverName'].unique(), starting_positions={})

        # Vérifier que tous les pilotes sont uniques
        if len(set(starting_positions.values())) != 22:
            error = f"Tous les pilotes doivent être uniques."
            return render_template('index.html', error=error, circuits=data['df']['circuitName'].unique(), pilotes=data['df']['driverName'].unique(), starting_positions=starting_positions)

        # Vérifier que tous les pilotes existent dans le dataframe
        df = data['df']
        all_pilots = df['driverName'].unique()
        not_found_pilots = [p for p in starting_positions.values() if p not in all_pilots]
        if not_found_pilots:
            error = f"Les pilotes suivants n'ont pas été trouvés dans la base de données : {', '.join(not_found_pilots)}"
            # Exclure les pilotes non trouvés de la liste des pilotes disponibles
            available_pilotes = [p for p in all_pilots if p not in starting_positions.values()]
            return render_template('index.html', error=error, circuits=data['df']['circuitName'].unique(), pilotes=available_pilotes, starting_positions=starting_positions)

        # Traiter les données pour la prédiction
        results = []

        # Encoder le circuit
        circuit_encoded = df[df['circuitName'] == circuit_name]['circuitId'].iloc[0]

        # Déterminer les conditions météorologiques
        high_temp = 1 if temp > 30 else 0
        strong_wind = 1 if wind_speed > 20 else 0

        for position, driver in starting_positions.items():
            driver_encoded = df[df['driverName'] == driver]['driverId'].iloc[0]

            # Préparer les données d'entrée pour le modèle
            input_data = [[
                circuit_encoded,
                df['avg_time_lap'].mean(),
                df['fastestLapTime'].mean(),
                1 if rain.lower() == 'yes' else 0,
                high_temp,
                strong_wind,
                temp,
                temp_track,
                humidity,
                wind_speed,
                df['WindDirection'].mean(),
                driver_encoded,
                position
            ]]

            input_data_df = pd.DataFrame(input_data, columns=data['X_train_columns'])
            input_data_scaled = data['scaler'].transform(input_data_df)
            position_pred = data['model'].predict(input_data_scaled)[0]
            results.append([driver, position, position_pred])

        # Trier les résultats par la position prédite
        results = sorted(results, key=lambda x: x[2])

        classement_final = []
        for idx, result in enumerate(results):
            classement_final.append({
                'Pilot': result[0],
                'Starting Position': result[1],
                'Estimated Position': idx + 1,
                'Score': round(result[2], 2)
            })

        # Calculer les métriques d'évaluation
        y_pred = data['model'].predict(data['X_test_scaled'])
        mae = mean_absolute_error(data['y_test'], y_pred)
        mse = mean_squared_error(data['y_test'], y_pred)
        ndcg_val = ndcg_score(np.asarray([data['y_test'].values.flatten()]), np.asarray([y_pred.flatten()]), k=10)

        return render_template('results.html', classement=classement_final, circuit_name=circuit_name, mae=mae, mse=mse, ndcg_val=ndcg_val)

    else:
        # GET request, rendre le formulaire avec les positions pré-remplies
        df = data['df']
        circuits = df['circuitName'].unique()
        all_pilotes = df['driverName'].unique()

        # Exemple de positions pré-remplies
        starting_positions = {


        }

        # Supprimer les pilotes déjà placés de la liste des pilotes disponibles
        available_pilotes = [pilot for pilot in all_pilotes if pilot not in starting_positions.values()]

        return render_template('index.html', circuits=circuits, pilotes=available_pilotes, starting_positions=starting_positions)

if __name__ == '__main__':
    app.run(debug=True)
