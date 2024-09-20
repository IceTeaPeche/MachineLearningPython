import pandas as pd

# Charger le fichier CSV
file_path = 'lap_times2.csv'  # Remplace par le chemin de ton fichier CSV
df = pd.read_csv(file_path)

# Vérifier les colonnes
print(df.columns)

# Convertir le format du temps en secondes : minutes + millisecondes
def convert_to_seconds(time, milliseconds):
    minutes, seconds = map(float, time.split(':'))  # Remplace 'time_str' par le nom correct si nécessaire
    total_time_seconds = minutes * 60 + seconds + milliseconds / 1000
    return total_time_seconds

# Appliquer la conversion (remplace 'time' par le nom correct si nécessaire)
df['total_time_seconds'] = df.apply(lambda row: convert_to_seconds(row['time'], row['milliseconds']), axis=1)

# Calculer la moyenne des temps pour chaque pilote par course
average_time_per_pilot = df.groupby(['driverId', 'raceId'])['total_time_seconds'].mean().reset_index()

# Trouver le tour le plus rapide pour chaque pilote
fastest_lap_per_pilot = df.groupby(['driverId', 'raceId'])['total_time_seconds'].min().reset_index()
fastest_lap_per_pilot.columns = ['driverId', 'raceId', 'fastest_lap_seconds']

# Fusionner les données pour avoir la moyenne et le tour le plus rapide dans le même DataFrame
df_final = pd.merge(average_time_per_pilot, fastest_lap_per_pilot, on=['driverId', 'raceId'])

# Sauvegarder le DataFrame final dans un fichier CSV
df_final.to_csv('lap_times_with_avg_and_fastest_lap.csv', index=False)

print("Les données ont été traitées avec succès et enregistrées dans 'lap_times_with_avg_and_fastest_lap.csv'.")
