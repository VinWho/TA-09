from flask import Flask, request, render_template, redirect, url_for, send_file, session
import pandas as pd
from datetime import datetime, timedelta
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import io
import base64

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Configuration
duration = timedelta(hours=1, minutes=30)
# start_date = datetime.strptime('01-03-2024', '%d-%m-%Y')
# end_date = datetime.strptime('30-07-2024', '%d-%m-%Y')
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
time_slots = ['08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00']
ruangan = ["GD Vokasi Lantai 1", "RMK"]  

global_fitness_over_time = []

@app.route('/')
def index():
    conflicts = []
    schedule = [] 
    return render_template('index.html', conflicts=conflicts, schedule=schedule, best_fitness=None, num_generations=None, pop_size=None, elapsed_time=None)

@app.route('/upload', methods=['POST'])
def upload_files():
    if request.method == 'POST':
        kelompok_file = request.files['kelompok']
        hari_file = request.files['hari']
        jadwal_dosen_file = request.files['jadwal_dosen']
        jadwal_mahasiswa_file = request.files['jadwal_mahasiswa']
        ruangan_file = request.files['ruangan']

        data_kel = pd.read_excel(kelompok_file, engine='openpyxl')
        hari_df = pd.read_excel(hari_file, engine='openpyxl')
        jadwal_dosen_df = pd.read_excel(jadwal_dosen_file, engine='openpyxl')
        jadwal_mahasiswa_df = pd.read_excel(jadwal_mahasiswa_file, engine='openpyxl')
        ruangan_df = pd.read_excel(ruangan_file, engine='openpyxl')

        ruangan_df.columns = ruangan_df.columns.str.strip().str.lower()

        global ruangan
        try:
            ruangan = list(ruangan_df['ruangan'])
        except KeyError:
            available_columns = ruangan_df.columns.tolist()
            return f"Error: 'ruangan' column not found in the uploaded file. Available columns: {available_columns}"

        # Simpan data ke sesi
        session['data_kel'] = data_kel.to_dict()
        session['ruangan'] = ruangan
        session['start_date'] = request.form['tanggal_mulai']
        session['end_date'] = request.form['tanggal_selesai']

        # Redirect setelah upload selesai
        return redirect(url_for('generate_new_schedule'))
    
    
    # Genetic Algorithm Functions
    #Inisialisasi Populasi
def initialize_population(pop_size, gene_pool, num_genes):
    population = []
    for _ in range(pop_size):
        individual = [random.randint(0, len(gene_pool) - 1) for _ in range(num_genes)] #memilih individu secara acak
        population.append(individual) #menambahkan individu ke populasi
    return population

def select_parents(population, fitnesses, num_parents, tournament_size=3):
    parents = []
    for _ in range(num_parents):
        # Pilih individu-individu secara acak untuk turnamen
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        # Pilih individu dengan fitness terbaik dari turnamen sebagai orang tua
        winner = max(tournament, key=lambda ind: ind[1])
        parents.append(winner[0])  # Tambahkan hanya individu 
    return parents


def crossover(parents, num_offspring, crossover_rate=0.8):
    offspring = [] #menyimpan keturunan yang dihasilkan
    for _ in range(num_offspring):
        if random.random() < crossover_rate: #menghasilkan nilai acak, dibandingkan ke crosover
            parent1, parent2 = random.sample(parents, 2) #pemilihan orang tua
            crossover_point = random.randint(1, len(parent1) - 1) #titik crossover
            child = parent1[:crossover_point] + parent2[crossover_point:] #membuat child
        else:
            child = random.choice(parents)  
        offspring.append(child) #jika tidak ada crossover ditambah ke offspring
    return offspring

def mutate(offspring, mutation_rate, gene_pool):
    for individual in offspring:
        for idx in range(len(individual)): #iterasi setiap gen
            if random.random() < mutation_rate: #cek mutasi
                individual[idx] = random.randint(0, len(gene_pool) - 1) #
    return offspring

def generate_all_possible_schedules(start_date, end_date, days, time_slots, ruangan):
    schedules = []
    current_date = start_date
    while current_date <= end_date:
        day_name = current_date.strftime('%A')
        if day_name in days: #cek daftar hari
            for room in ruangan:
                for start_time in time_slots:
                    end_time = (datetime.strptime(start_time, '%H:%M') + duration).strftime('%H:%M')
                    schedules.append((current_date.strftime('%d-%m-%Y'), day_name, start_time, end_time, room))
        current_date += timedelta(days=1)
    return schedules

def convert_to_datetime(time_slot):
    start_time, end_time = time_slot.split('-')
    start_dt = datetime.strptime(start_time.strip(), '%H:%M')
    end_dt = datetime.strptime(end_time.strip(), '%H:%M')
    return start_dt, end_dt

def is_time_overlap(slot1, slot2):
    start1, end1 = slot1
    start2, end2 = slot2
    return max(start1, start2) < min(end1, end2)

def fitness_func(solution, all_schedules, data_kel, student_schedule, lecturer_schedule):
    conflicts = 0
    conflict_details = []  # List to store conflict details
    num_schedules = len(solution)

    for idx, schedule in enumerate(solution):
        date, day, start_time, end_time, room = all_schedules[schedule]
        start_time, end_time = convert_to_datetime(f"{start_time}-{end_time}")
        kelompok = data_kel.iloc[idx] #ambil data kelompok
        
        involved_lecturers = [
            kelompok['Dosen_Pembimbing_1'], kelompok['Dosen_Pembimbing_2'],
            kelompok['Dosen_Penguji_1'], kelompok['Dosen_Penguji_2']
        ] #ambil daftar dosen dalam kelompok
        involved_lecturers = [lecturer for lecturer in involved_lecturers if lecturer] #filter daftar dosen -> nan

        # Check room conflicts
        for other_idx, other_schedule in enumerate(solution): #membandingkan jadwal saat ini dengan jadwal lain cek konflik
            if idx != other_idx: #pastikan mebandingkan dengan jadwal lain
                other_date, other_day, other_start_time, other_end_time, other_room = all_schedules[other_schedule]
                other_start, other_end = convert_to_datetime(f"{other_start_time}-{other_end_time}")
                if date == other_date and room == other_room: #cek waktu dari jadwal seminar lain
                    if is_time_overlap((start_time, end_time), (other_start, other_end)):
                        conflicts += 1
                        conflict_details.append({
                            'kelompok1': kelompok['Kelompok'],
                            'kelompok2': data_kel.iloc[other_idx]['Kelompok'],
                            'conflict_type': 'Room Conflict',
                            'details': f"Both scheduled in {room} on {date} from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}"
                        })

        # Check student conflicts
        #constraint 5
        if day in student_schedule: #cek hari yg diperiksa
            for student_time in student_schedule[day]: #iterasiy
                student_start, student_end = convert_to_datetime(student_time)
                if is_time_overlap((start_time, end_time), (student_start, student_end)):
                    conflicts += 1
                    conflict_details.append({
                        'kelompok': kelompok['Kelompok'],
                        'conflict_type': 'Student Conflict',
                        'details': f"Scheduled during student class time on {date} from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}"
                    })

        # Check lecturer conflicts with teaching schedule
        for lecturer in involved_lecturers:
        #constraint 4
            if day in lecturer_schedule and lecturer in lecturer_schedule[day]:
                for teaching_time in lecturer_schedule[day][lecturer]:
                    teaching_start, teaching_end = convert_to_datetime(teaching_time)
                    if is_time_overlap((start_time, end_time), (teaching_start, teaching_end)):
                        conflicts += 1
                        conflict_details.append({
                            'kelompok': kelompok['Kelompok'],
                            'lecturer': lecturer,
                            'conflict_type': 'Lecturer Teaching Conflict',
                            'details': f"{lecturer} is teaching on {date} from {teaching_start.strftime('%H:%M')} to {teaching_end.strftime('%H:%M')}"
                        })

        # Check lecturer conflicts within seminars
        for other_idx, other_schedule in enumerate(solution):
            if idx != other_idx: #memastikan membandingkan jadwal yg berbeda untuk cek konflik
                other_date, other_day, other_start_time, other_end_time, other_room = all_schedules[other_schedule]
                other_start, other_end = convert_to_datetime(f"{other_start_time}-{other_end_time}")
                other_kelompok = data_kel.iloc[other_idx] #mengambil data kelompok lain pada index
                other_involved_lecturers = [
                    other_kelompok['Dosen_Pembimbing_1'], other_kelompok['Dosen_Pembimbing_2'],
                    other_kelompok['Dosen_Penguji_1'], other_kelompok['Dosen_Penguji_2']
                ] #menyusun daftar dosen yang mungkin terlibat di konflik
                other_involved_lecturers = [lecturer for lecturer in other_involved_lecturers if lecturer] #menyaring daftar hanya nan 

                #constraint 1
                for lecturer in involved_lecturers:
                    if lecturer in other_involved_lecturers: #cek dosen di jadwal 
                        if date == other_date and is_time_overlap((start_time, end_time), (other_start, other_end)): #cek jadwal tumpang tindih atau tidak dari jadwal seminar kel lain
                            if lecturer in [kelompok['Dosen_Pembimbing_1'], kelompok['Dosen_Pembimbing_2']] and lecturer in [other_kelompok['Dosen_Penguji_1'], other_kelompok['Dosen_Penguji_2']]: #cek dosen pem menguji di kel lain
                                conflicts += 1
                                conflict_details.append({
                                    'kelompok1': kelompok['Kelompok'],
                                    'kelompok2': other_kelompok['Kelompok'],
                                    'lecturer': lecturer,
                                    'conflict_type': 'Supervisor as Examiner Conflict',
                                    'details': f"{lecturer} has a conflict on {date} from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}"
                                })
                            #constraint 2
                            elif lecturer in [kelompok['Dosen_Penguji_1'], kelompok['Dosen_Penguji_2']] and lecturer in [other_kelompok['Dosen_Penguji_1'], other_kelompok['Dosen_Penguji_2']]:#memeriksa apakah dosen tertentu jadi penguji di dua kelompok yang sama pada saat yang sama
                                conflicts += 1
                                conflict_details.append({
                                    'kelompok1': kelompok['Kelompok'],
                                    'kelompok2': other_kelompok['Kelompok'],
                                    'lecturer': lecturer,
                                    'conflict_type': 'Examiner Conflict',
                                    'details': f"{lecturer} has a conflict on {date} from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}"
                                })
                            #constraint 3
                            elif lecturer in [kelompok['Dosen_Pembimbing_1'], kelompok['Dosen_Pembimbing_2']] and lecturer in [other_kelompok['Dosen_Pembimbing_1'], other_kelompok['Dosen_Pembimbing_2']]:
                                conflicts += 1
                                conflict_details.append({
                                    'kelompok1': kelompok['Kelompok'],
                                    'kelompok2': other_kelompok['Kelompok'],
                                    'lecturer': lecturer,
                                    'conflict_type': 'Supervisor Conflict',
                                    'details': f"{lecturer} has a conflict on {date} from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}"
                                })

    # Calculate the maximum number of possible conflicts
    max_conflicts = (num_schedules * (num_schedules - 1)) * 3 
    
    # Ensure max_conflicts is non-zero to avoid division by zero
    if max_conflicts == 0:
        return 1.0, conflict_details  

    # Calculate fitness as a value between 0 and 1
    fitness = 1 - (conflicts / max_conflicts)
    return fitness, conflict_details

def run_genetic_algorithm(data_kel, student_schedule, lecturer_schedules, ruangan, start_date, end_date):
    global global_fitness_over_time  
    global_fitness_over_time = []  # Initialize or reset before starting
    
    all_schedules = generate_all_possible_schedules(start_date, end_date, days, time_slots, ruangan)
    num_genes = len(data_kel)

    pop_size = 50
    num_generations = 100
    num_parents = 5
    mutation_rate = 0.1
    crossover_rate = 0.8

    population = initialize_population(pop_size, all_schedules, num_genes)

    start_time = time.time()    

    fitness_history = []
    fitness_scores = []  
    best_conflict_details = []

    for generation in range(num_generations):
        fitnesses = []
        conflict_details_population = []

        for individual in population:
            fitness, conflict_details = fitness_func(individual, all_schedules, data_kel, student_schedule, lecturer_schedules)
            fitnesses.append(fitness)
            conflict_details_population.append(conflict_details)

        best_fitness_idx = fitnesses.index(max(fitnesses))
        fitness_history.append(max(fitnesses))
        global_fitness_over_time.append(max(fitnesses))  # Store the best fitness for each generation

        print(f"Generation {generation}: Best Fitness = {max(fitnesses)}")

        if max(fitnesses) == 1:
            break

        parents = select_parents(population, fitnesses, num_parents) #select parents
        offspring = crossover(parents, pop_size - len(parents), crossover_rate)
        offspring = mutate(offspring, mutation_rate, all_schedules)

        population = parents + offspring #gabungkan parents dan child

    elapsed_time = time.time() - start_time

    if not os.path.exists('static'):
        os.makedirs('static')

    plt.figure()
    plt.plot(fitness_history, label='Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    fitness_plot_path = 'static/fitness_plot.png'
    plt.savefig(fitness_plot_path)
    plt.close()

    best_solution = population[fitnesses.index(max(fitnesses))]

    best_schedule = [(data_kel.iloc[idx], all_schedules[schedule]) for idx, schedule in enumerate(best_solution)]
    schedule_data = []
    for kelompok, (date, day, start_time, end_time, room) in best_schedule:
        combined_time = f"{start_time}-{end_time}"
        schedule_data.append({
            'Kelompok': kelompok['Kelompok'],
            'Dosen_Penguji_1': kelompok['Dosen_Penguji_1'],
            'Dosen_Penguji_2': kelompok['Dosen_Penguji_2'],
            'Dosen_Pembimbing_1': kelompok['Dosen_Pembimbing_1'],
            'Dosen_Pembimbing_2': kelompok['Dosen_Pembimbing_2'],
            'Hari': day,
            'Tanggal': date,
            'Waktu': combined_time,  # Combined time in one cell
            'Ruangan': room
        })
    
    schedule_df = pd.DataFrame(schedule_data)

    best_fitness = max(fitnesses)
    accuracy = calculate_accuracy(conflict_details_population[best_fitness_idx], len(data_kel), best_fitness)

    return schedule_df, best_fitness, num_generations, pop_size, elapsed_time, fitness_plot_path, conflict_details_population[best_fitness_idx], accuracy


def calculate_accuracy(conflict_details, total_seminars, best_fitness):
    seminars_with_conflicts = len(set([conflict['kelompok1'] for conflict in conflict_details]))
    seminars_without_conflicts = total_seminars - seminars_with_conflicts
    # accuracy = (best_fitness * 100)  
    accuracy = (seminars_without_conflicts / total_seminars)*100
    return accuracy


@app.route('/generate_new_schedule')
def generate_new_schedule():
    # Ambil data dari sesi
    data_kel = pd.DataFrame(session['data_kel'])
    ruangan = session['ruangan']
    start_date = datetime.strptime(session['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(session['end_date'], '%Y-%m-%d')

    student_schedule = {}
    lecturer_schedules = {}

    schedule_df, best_fitness, num_generations, pop_size, elapsed_time, fitness_plot_path, conflict_details, best_accuracy = run_genetic_algorithm(
        data_kel, student_schedule, lecturer_schedules, ruangan, start_date, end_date
    )
    
    # Simpan jadwal ke dalam file Excel
    filename = os.path.join(os.getcwd(), "jadwal_seminar.xlsx")
    schedule_df.to_excel(filename, index=False)  # Simpan ke file Excel

    schedule = schedule_df.to_dict(orient='records')
    return render_template('index.html', conflicts=conflict_details, schedule=schedule, best_fitness=best_fitness, num_generations=num_generations, pop_size=pop_size, elapsed_time=elapsed_time, best_accuracy=best_accuracy)

@app.route("/download_schedule")
def download_schedule():
    filename = os.path.join(os.getcwd(), "jadwal_seminar.xlsx")
    
    if not os.path.exists(filename):
        return "File not found!", 404
    
    return send_file(filename, as_attachment=True)


@app.route("/plot_fitness")
def plot_fitness():
    global global_fitness_over_time

    if not global_fitness_over_time:
        return "Tidak ada data untuk ditampilkan", 400

    plt.figure()
    plt.plot(global_fitness_over_time)
    plt.xlabel("Generasi")
    plt.ylabel("Fitness")
    plt.title("Fitness Selama Generasi")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return f'<img src="data:image/png;base64,{plot_url}"/>'


if __name__ == '__main__':
    app.run(debug=True)
