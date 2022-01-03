# -*- coding: utf-8 -*-
"""
Training for PGDL model
@author: xiaoweijia
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
import feather
import argparse
import os
import pandas
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='fig_1/tmp/mendota/train/inputs')
parser.add_argument('--restore_path')  # default='fig_1/tmp/mendota/pretrain/model'
parser.add_argument('--save_path', default='fig_1/tmp/mendota/train/model')
parser.add_argument('--preds_path', default='fig_1/tmp/mendota/train/out')
args = parser.parse_args()

# tf.reset_default_graph()
random.seed(9001)

''' Declare constant hyperparameters '''
learning_rate = 0.05
epochs = 250 
state_size = 20  # net configuration
input_size = 9
phy_size = 10  # physics info size
n_steps = 352  # window size
n_classes = 1  # output size
N_sec = 19
elam = 1
ec_threshold = 24
if_PINN = True

''' Define Graph '''
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder("float", [None, n_steps, input_size])
y = tf.compat.v1.placeholder("float", [None, n_steps])
m = tf.compat.v1.placeholder("float", [None, n_steps])
bt_sz = tf.compat.v1.placeholder("int32", None)
x_u = tf.compat.v1.placeholder("float", [None, n_steps, input_size])

lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(state_size, forget_bias=1.0)

state_series_x, current_state_x = tf.compat.v1.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
w_fin = tf.compat.v1.get_variable('w_fin', [state_size, n_classes], tf.float32,
                                  tf.compat.v1.initializers.random_normal(stddev=0.02), use_resource=False)
b_fin = tf.compat.v1.get_variable('b_fin', [n_classes], tf.float32, initializer=tf.compat.v1.initializers.constant(0.0),
                                  use_resource=False)

pred = []
for i in range(n_steps):
    tp1 = state_series_x[:, i, :]
    pt = tf.matmul(tp1, w_fin) + b_fin
    pred.append(pt)

pred = tf.stack(pred, axis=1)
pred_s = tf.reshape(pred, [-1, 1])
y_s = tf.reshape(y, [-1, 1])
m_s = tf.reshape(m, [-1, 1])

raw_cost = tf.sqrt(
    tf.reduce_sum(input_tensor=tf.square(tf.multiply((pred_s - y_s), m_s))) / tf.reduce_sum(input_tensor=m_s))

''' Define Physics Helper Functions '''


def transformTempToDensity(temp):
    densities = 1000 * (1 - ((temp + 288.9414) * tf.pow(temp - 3.9863, 2)) / (508929.2 * (temp + 68.12963)))
    return densities


def calculate_ec_loss(inputs, outputs, phys, depth_areas, n_depths, ec_threshold, combine_days=1):
    densities = transformTempToDensity(outputs)

    diff_per_set = []
    # loop through sets of n_depths
    for i in range(N_sec):
        # indices
        # (n_depths = depth_areas.size)
        start_index = (i) * n_depths
        end_index = (i + 1) * n_depths

        # calculate lake energy for each timestep
        lake_energies = calculate_lake_energy(
            outputs[start_index:end_index, :], densities[start_index:end_index, :], depth_areas)

        # calculate energy change in each timestep
        lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
        lake_energy_deltas = lake_energy_deltas[1:]

        # calculate sum of energy flux into or out of the lake at each timestep
        lake_energy_fluxes = calculate_energy_fluxes(phys[start_index, :, :], outputs[start_index, :], combine_days)
        diff_vec = tf.abs(lake_energy_deltas - lake_energy_fluxes)

        # ice mask
        tmp_mask = 1 - phys[start_index + 1, 1:-1, 9]
        tmp_loss = tf.reduce_mean(input_tensor=diff_vec * tf.cast(tmp_mask, tf.float32))
        diff_per_set.append(tmp_loss)

    diff_per_set_r = tf.stack(diff_per_set)

    diff_per_set = tf.clip_by_value(diff_per_set_r - ec_threshold, clip_value_min=0, clip_value_max=999999)
    # TODO
    # temps [-1, n_steps(352)], phys [None, n_steps(352), phy_size(10)]
    depth_loss = []
    for i in range(n_steps):
        depth_loss.append(calculate_depth_loss(outputs[:, i], phys[:, i, :]))
    # diff_per_set = tf.reduce_mean(input_tensor=diff_per_set) + tf.reduce_mean(input_tensor=depth_loss)
    diff_per_set = tf.reduce_mean(input_tensor=depth_loss)

    if if_PINN:
        return diff_per_set, tf.reduce_mean(input_tensor=depth_loss), diff_per_set_r, diff_per_set
    else:
        return tf.zeros_like(diff_per_set), tf.zeros_like(diff_vec), tf.zeros_like(
            diff_per_set_r), tf.zeros_like(diff_per_set)


def calculate_depth_loss(temps, depth):
    # for i in range(n_depths):
    #     x_raw_full[i, :, 1] = i * 0.5  # fill in the depth column as depth in m (0, 0.5, 1, ..., (n_depths-1)/2)
    # The deeper the colder (closer to 4C ???)
    depth_loss = 0
    depth1 = depth[:-1, 1]
    depth2 = depth[1:, 1]
    temps1 = temps[:-1]
    temps2 = temps[1:]
    mask = tf.math.less_equal(depth2, depth1, name=None)  # depth1[i] < depth2[i]
    prim_loss = tf.sigmoid(10*(temps1 - temps2))
    prim_loss = tf.boolean_mask(prim_loss, mask)
    depth_loss = depth_loss + prim_loss
    return depth_loss


def calculate_lake_energy(temps, densities, depth_areas):
    # calculate the total energy of the lake for every timestep
    # sum over all layers the (depth cross-sectional area)*temp*density*layer_height)
    # then multiply by the specific heat of water
    dz = 0.5  # thickness for each layer
    cw = 4186  # specific heat of water
    depth_areas = tf.reshape(depth_areas, [n_depths, 1])
    energy = tf.reduce_sum(input_tensor=tf.multiply(tf.cast(depth_areas, tf.float32), temps) * densities * dz * cw,
                           axis=0)
    return energy


def calculate_lake_energy_deltas(energies, combine_days, surface_area):
    # given a time series of energies, compute and return the differences
    # between each time step
    time = 86400  # seconds per day
    energy_deltas = (energies[1:] - energies[:-1]) / time / surface_area
    return energy_deltas


def calculate_air_density(air_temp, rh):
    # returns air density in kg / m^3
    # equation from page 13 GLM/GLEON paper(et al Hipsey)

    # Ratio of the molecular (or molar) weight of water to dry air
    mwrw2a = 18.016 / 28.966
    c_gas = 1.0e3 * 8.31436 / 28.966

    # atmospheric pressure
    p = 1013.  # mb

    # water vapor pressure
    vapPressure = calculate_vapour_pressure_air(rh, air_temp)

    # water vapor mixing ratio (from GLM code glm_surface.c)
    r = mwrw2a * vapPressure / (p - vapPressure)
    return (1.0 / c_gas * (1 + r) / (1 + r / mwrw2a) * p / (air_temp + 273.15)) * 100


def calculate_heat_flux_sensible(surf_temp, air_temp, rel_hum, wind_speed):
    # equation 22 in GLM/GLEON paper(et al Hipsey)
    # GLM code ->  Q_sensibleheat = -CH * (rho_air * 1005.) * WindSp * (Lake[surfLayer].Temp - MetData.AirTemp);
    # calculate air density
    rho_a = calculate_air_density(air_temp, rel_hum)

    # specific heat capacity of air in J/(kg*C)
    c_a = 1005.

    # bulk aerodynamic coefficient for sensible heat transfer
    c_H = 0.0013

    # wind speed at 10m
    U_10 = calculate_wind_speed_10m(wind_speed)

    return -rho_a * c_a * c_H * U_10 * (surf_temp - air_temp)


def calculate_heat_flux_latent(surf_temp, air_temp, rel_hum, wind_speed):
    # equation 23 in GLM/GLEON paper(et al Hipsey)
    # GLM code-> Q_latentheat = -CE * rho_air * Latent_Heat_Evap * (0.622/p_atm) * WindSp * (SatVap_surface - MetData.SatVapDef)
    # where,         SatVap_surface = saturated_vapour(Lake[surfLayer].Temp);
    #                rho_air = atm_density(p_atm*100.0,MetData.SatVapDef,MetData.AirTemp);
    # air density in kg/m^3
    rho_a = calculate_air_density(air_temp, rel_hum)

    # bulk aerodynamic coefficient for latent heat transfer
    c_E = 0.0013

    # latent heat of vaporization (J/kg)
    lambda_v = 2.453e6

    # wind speed at 10m height
    U_10 = calculate_wind_speed_10m(wind_speed)

    # ratio of molecular weight of water to that of dry air
    omega = 0.622

    # air pressure in mb
    p = 1013.

    e_s = calculate_vapour_pressure_saturated(surf_temp)
    e_a = calculate_vapour_pressure_air(rel_hum, air_temp)
    return -rho_a * c_E * lambda_v * U_10 * (omega / p) * (e_s - e_a)


def calculate_vapour_pressure_air(rel_hum, temp):
    rh_scaling_factor = 1
    return rh_scaling_factor * (rel_hum / 100) * calculate_vapour_pressure_saturated(temp)


def calculate_vapour_pressure_saturated(temp):
    # returns in millibars
    exponent = (9.28603523 - (2332.37885 / (temp + 273.15))) * np.log(10)
    return tf.exp(exponent)


def calculate_wind_speed_10m(ws, ref_height=2.):
    # from GLM code glm_surface.c
    c_z0 = 0.001  # default roughness
    return ws * (tf.math.log(10.0 / c_z0) / tf.math.log(ref_height / c_z0))


def calculate_energy_fluxes(phys, surf_temps, combine_days):
    e_s = 0.985  # emissivity of water
    alpha_sw = 0.07  # shortwave albedo
    alpha_lw = 0.03  # longwave albedo
    sigma = 5.67e-8  # Stefan-Baltzmann constant
    R_sw_arr = phys[:-1, 2] + (phys[1:, 2] - phys[:-1, 2]) / 2
    R_lw_arr = phys[:-1, 3] + (phys[1:, 3] - phys[:-1, 3]) / 2
    R_lw_out_arr = e_s * sigma * (tf.pow(surf_temps[:] + 273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:] - R_lw_out_arr[:-1]) / 2

    air_temp = phys[:-1, 4]
    air_temp2 = phys[1:, 4]
    rel_hum = phys[:-1, 5]
    rel_hum2 = phys[1:, 5]
    ws = phys[:-1, 6]
    ws2 = phys[1:, 6]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2) / 2
    H = (H + H2) / 2

    fluxes = (R_sw_arr[:-1] * (1 - alpha_sw) + R_lw_arr[:-1] * (1 - alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])
    return fluxes


''' Continue Graph Definition '''

unsup_inputs = tf.compat.v1.placeholder("float", [None, n_steps, input_size])

with tf.compat.v1.variable_scope("rnn", reuse=True) as scope_sp:
    state_series_xu, current_state_xu = tf.compat.v1.nn.dynamic_rnn(lstm_cell, unsup_inputs, dtype=tf.float32,
                                                                    scope=scope_sp)

pred_u = []
for i in range(n_steps):
    tp2 = state_series_xu[:, i, :]
    pt2 = tf.matmul(tp2, w_fin) + b_fin
    pred_u.append(pt2)

pred_u = tf.stack(pred_u, axis=1)
pred_u = tf.reshape(pred_u, [-1, n_steps])

unsup_phys_data = tf.compat.v1.placeholder("float", [None, n_steps, phy_size])  # tf.float32

depth_areas = np.load(os.path.join(args.data_path, 'depth_areas.npy'))
n_depths = depth_areas.size

unsup_loss, a, b, c = calculate_ec_loss(unsup_inputs,
                                        pred_u,
                                        unsup_phys_data,
                                        depth_areas,
                                        n_depths,
                                        ec_threshold,
                                        combine_days=1)

cost = raw_cost + elam * unsup_loss

tvars = tf.compat.v1.trainable_variables()
for i in tvars:
    print(i)
grads = tf.gradients(ys=cost, xs=tvars)

saver = tf.compat.v1.train.Saver(max_to_keep=5)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(grads, tvars))

''' Load data '''

x_full = np.load(os.path.join(args.data_path, 'processed_features.npy'))
x_raw_full = np.load(os.path.join(args.data_path, 'features.npy'))
diag_full = np.load(os.path.join(args.data_path, 'diag.npy'))

# for i in range(n_depths):
#    x_raw_full[i, :, 1] = i * 0.5  # fill in the depth column as depth in m (0, 0.5, 1, ..., (n_depths-1)/2)

# ['DOY', 'depth', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Daily.Qe', 'Daily.Qh', 'Has.Black.Ice']
phy_full = np.concatenate((x_raw_full[:, :, :-2], diag_full), axis=2)

new_dates = np.load(os.path.join(args.data_path, 'dates.npy'), allow_pickle=True)

train_data = feather.read_dataframe(os.path.join(args.data_path, 'labels_train.feather'))

tr_date = train_data.values[:, 0]
tr_depth = train_data.values[:, 1]
tr_temp = train_data.values[:, 2]

t_steps = x_raw_full.shape[1]
m_tr = np.zeros([n_depths, t_steps])
obs_tr = np.zeros([n_depths, t_steps])
k = 0
# dd = 0
for i in range(new_dates.shape[0]):
    if k >= tr_date.shape[0]:
        break
    while new_dates[i] == tr_date[k]:
        d = min(int(tr_depth[k] / 0.5), n_depths - 1)
        m_tr[d, i] = 1
        obs_tr[d, i] = tr_temp[k]
        k += 1
        if k >= tr_date.shape[0]:
            break

test_data = feather.read_dataframe(os.path.join(args.data_path, 'labels_test.feather'))

te_date = test_data.values[:, 0]
te_depth = test_data.values[:, 1]
te_temp = test_data.values[:, 2]

m_te = np.zeros([n_depths, t_steps])
obs_te = np.zeros([n_depths, t_steps])
k = 0
# dd = 0
for i in range(new_dates.shape[0]):
    if k >= te_date.shape[0]:
        break
    while new_dates[i] == te_date[k]:
        d = min(int(te_depth[k] / 0.5), n_depths - 1)
        # if m_te[d, i] == 1:
        #   print(d, te_depth[k])
        m_te[d, i] = 1
        obs_te[d, i] = te_temp[k]
        k += 1
        if k >= te_date.shape[0]:
            break

x_train = np.zeros([n_depths * N_sec, n_steps, input_size])
y_train = np.zeros([n_depths * N_sec, n_steps])
phy_train = np.zeros([n_depths * N_sec, n_steps, phy_size])
m_train = np.zeros([n_depths * N_sec, n_steps])
y_test = np.zeros([n_depths * N_sec, n_steps])
m_test = np.zeros([n_depths * N_sec, n_steps])

for i in range(1, N_sec + 1):
    x_train[(i - 1) * n_depths:i * n_depths, :, :] = x_full[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2), :]
    y_train[(i - 1) * n_depths:i * n_depths, :] = obs_tr[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    phy_train[(i - 1) * n_depths:i * n_depths, :, :] = phy_full[:,
                                                       int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2),
                                                       :]
    m_train[(i - 1) * n_depths:i * n_depths, :] = m_tr[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]

    y_test[(i - 1) * n_depths:i * n_depths, :] = obs_te[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    m_test[(i - 1) * n_depths:i * n_depths, :] = m_te[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]

x_f = x_train
phy_f = phy_train

''' Train '''
epoches = []
losses = []
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    # If using pretrained model, reload it now
    # if args.restore_path != '':
    #    saver.restore(sess, os.path.join(args.restore_path, 'pretrained_model.ckpt'))

    for epoch in range(epochs):
        _, loss, rc, ec, aa, bb, cc, prd1 = sess.run(
            [train_op, cost, raw_cost, unsup_loss, a, b, c, pred],
            feed_dict={
                x: x_train,
                y: y_train,
                m: m_train,
                unsup_inputs: x_f,
                unsup_phys_data: phy_f,
                bt_sz: n_depths * N_sec
            })

        if epoch % 1 == 0:
            loss_val = sess.run(raw_cost, feed_dict={x: x_train, y: y_test, m: m_test, bt_sz: n_depths * N_sec})
            print("Epoch " + str(epoch) + ", BatLoss= " + \
                  "{:.4f}".format(loss) + ", MSE= " + \
                  "{:.4f}".format(rc) + ", Ec= " + \
                  "{:.4f}".format(ec) + ", Validation loss= " + \
                  "{:.4f}".format(loss_val))
            epoches.append(int(epoch))
            losses.append(loss_val)

    loss_te, prd = sess.run([raw_cost, pred], feed_dict={x: x_train, y: y_test, m: m_test, bt_sz: n_depths * N_sec})

    print("Loss_te " + "{:.4f}".format(loss_te))
    plt.plot(epoches, losses)
    plt.show()
    if not if_PINN:
        np.savetxt('nepoches.csv', epoches, delimiter=',')
        np.savetxt('nlosses.csv', losses, delimiter=',')
        epoches = np.loadtxt('epoches.csv', delimiter=',')
        losses = np.loadtxt('losses.csv', delimiter=',')
    else:
        np.savetxt('epoches.csv', epoches, delimiter=',')
        np.savetxt('losses.csv', losses, delimiter=',')
        nepoches = np.loadtxt('nepoches.csv', delimiter=',')
        nlosses = np.loadtxt('nlosses.csv', delimiter=',')

    plt.plot(epoches, losses, label='PINN')
    plt.plot(nepoches, nlosses, label='non_PINN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    if args.save_path != '':
        saver.save(sess, os.path.join(args.save_path, "trained_model.ckpt"))

    # predict on test data, reshape to output file format, and save
    loss_te, prd = sess.run([raw_cost, pred], feed_dict={x: x_train, y: y_test, m: m_test})
    prd_o = np.zeros([n_depths, n_steps + int((N_sec - 1) * n_steps / 2)])
    prd_o[:, :n_steps] = prd[0:n_depths, :, 0]
    for j in range(N_sec - 1):
        st_idx = n_steps - (int((j + 1) * n_steps / 2) - int(j * n_steps / 2))  # handle even or odd cases
        prd_o[:, n_steps + int(j * n_steps / 2):n_steps + int((j + 1) * n_steps / 2)] = prd[(j + 1) * n_depths:(
                                                                                                                       j + 2) * n_depths,
                                                                                        st_idx:, 0]
    np.savetxt(os.path.join(args.preds_path, "predict_pgdl.csv"), prd_o, delimiter=',')

"""
## python ##
import re
import sciencebasepy
import os
download = False
if download:
    raw_data_path = 'fig_1/tmp/mendota/shared/raw_data'
    pretrain_inputs_path = 'fig_1/tmp/mendota/pretrain/inputs'
    pretrain_model_path = 'fig_1/tmp/mendota/pretrain/model'
    train_inputs_path = 'fig_1/tmp/mendota/train/similar_980_1/inputs'
    train_model_path = 'fig_1/tmp/mendota/train/similar_980_1/model'
    predictions_path = 'fig_1/tmp/mendota/train/similar_980_1/out'
    if not os.path.isdir(raw_data_path): os.makedirs(raw_data_path)

    if not os.path.isdir(pretrain_inputs_path): os.makedirs(pretrain_inputs_path)

    if not os.path.isdir(pretrain_model_path): os.makedirs(pretrain_model_path)

    if not os.path.isdir(train_inputs_path): os.makedirs(train_inputs_path)

    if not os.path.isdir(train_model_path): os.makedirs(train_model_path)

    if not os.path.isdir(predictions_path): os.makedirs(predictions_path)

    # Configure access to ScienceBase access
    sb = sciencebasepy.SbSession()
    # Th following line should only be necessary before the data release is public:
    # sb.login('[username]', '[password]') # manually revise username and password

    def download_from_sciencebase(item_id, search_text, to_folder):
        item_info = sb.get_item(item_id)
        file_info = [f for f in item_info['files'] if re.search(search_text, f['name'])][0]
        sb.download_file(file_info['downloadUri'], file_info['name'], to_folder)
        return os.path.join(to_folder, file_info['name'])

    # Data release URLs can be browsed by adding one of the following IDs after "https://www.sciencebase.gov/catalog/item/",
    # e.g., https://www.sciencebase.gov/catalog/item/5d98e0c4e4b0c4f70d1186f1
    met_file = download_from_sciencebase('5d98e0c4e4b0c4f70d1186f1', 'meteo.csv', raw_data_path)
    ice_file = download_from_sciencebase('5d98e0c4e4b0c4f70d1186f1', 'pretrainer_ice_flags.csv', raw_data_path)
    glm_file = download_from_sciencebase('5d915cb2e4b0c4f70d0ce523', 'predict_pb0.csv', raw_data_path)
    train_obs_file = download_from_sciencebase('5d8a837fe4b0c4f70d0ae8ac', 'similar_training.csv', raw_data_path)
    test_obs_file = download_from_sciencebase('5d925066e4b0c4f70d0d0599', 'test.csv', raw_data_path)


import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['pretrain', 'train'], default='train')
parser.add_argument('--lake_name', choices=['mendota', 'sparkling'], default='mendota')
parser.add_argument('--met_file', default='fig_1/tmp/mendota/shared/raw_data/mendota_meteo.csv')
parser.add_argument('--glm_file', default='fig_1/tmp/mendota/shared/raw_data/me_predict_pb0.csv')
parser.add_argument('--ice_file', default='fig_1/tmp/mendota/shared/raw_data/mendota_pretrainer_ice_flags.csv')
parser.add_argument('--processed_path', default='fig_1/tmp/mendota/train/similar_980_1/inputs')
args = parser.parse_args()

# Define and save some depth-area relationship and other lake-specific tidbits
if args.lake_name == 'mendota':
    depth_areas = np.array([
        39865825, 38308175, 38308175, 35178625, 35178625, 33403850, 31530150, 31530150, 30154150, 30154150, 29022000,
        29022000, 28063625, 28063625, 27501875, 26744500, 26744500, 26084050, 26084050, 25310550, 24685650, 24685650,
        23789125, 23789125, 22829450, 22829450, 21563875, 21563875, 20081675, 18989925, 18989925, 17240525, 17240525,
        15659325, 14100275, 14100275, 12271400, 12271400, 9962525, 9962525, 7777250, 7777250, 5956775, 4039800, 4039800,
        2560125, 2560125, 820925, 820925, 216125])
    data_chunk_size = 5295  # size of half the dates in the pretraining period
elif args.lake_name == 'sparkling':
    depth_areas = np.array([
        637641.569, 637641.569, 592095.7426, 592095.7426, 546549.9163, 546549.9163, 546549.9163, 501004.0899,
        501004.0899, 501004.0899, 455458.2636, 455458.2636, 409912.4372, 409912.4372, 409912.4372, 364366.6109,
        364366.6109, 318820.7845, 318820.7845, 318820.7845, 273274.9581, 273274.9581, 273274.9581, 227729.1318,
        227729.1318, 182183.3054, 182183.3054, 182183.3054, 136637.4791, 136637.4791, 136637.4791, 91091.65271,
        91091.65271, 45545.82636, 45545.82636, 45545.82636])
    data_chunk_size = 5478
n_depths = depth_areas.size
np.save(os.path.join(args.processed_path, 'depth_areas.npy'), depth_areas)
np.save(os.path.join(args.processed_path, 'data_chunk_size.npy'), data_chunk_size)

feat = pd.read_csv(args.met_file)  # features (meteorological drivers)
glm = pd.read_csv(args.glm_file)  # GLM predictions

if args.lake_name == 'sparkling':
    glm = glm.drop(columns='temp_18')

# Truncate to the training or testing period
if args.phase == 'pretrain':
    feat = feat[pd.to_datetime(feat['date'].values) <= pd.to_datetime('2009-04-01')]
elif args.phase == 'train':
    feat = feat[pd.to_datetime(feat['date'].values) > pd.to_datetime('2009-04-01')]

feat = feat.merge(glm[['date']], on='date')
glm = glm.merge(feat[['date']], on='date')

# create dates, x_full, x_raw_full, diag_full, label(glm)
x_raw_full = feat.drop('date',
                       axis=1).values  # ['ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow']
new_dates = feat[['date']].values[:, 0]
np.save(os.path.join(args.processed_path, 'dates.npy'), new_dates)

n_steps = x_raw_full.shape[0]

import datetime

format = "%Y-%m-%d"

doy = np.zeros([n_steps, 1])
for i in range(n_steps):
    dt = datetime.datetime.strptime(str(new_dates[i]), format)
    tt = dt.timetuple()
    doy[i, 0] = tt.tm_yday

x_raw_full = np.concatenate([doy, np.zeros([n_steps, 1]), x_raw_full],
                            axis=1)  # ['DOY', 'depth', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow']
x_raw_full = np.tile(x_raw_full, [n_depths, 1, 1])  # add depth replicates as prepended first dimension

for i in range(n_depths):
    x_raw_full[i, :, 1] = i * 0.5  # fill in the depth column as depth in m (0, 0.5, 1, ..., (n_depths-1)/2)

# copy into matrix, still with columns ['DOY', 'depth', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow']
x_raw_full_new = np.zeros([x_raw_full.shape[0], x_raw_full.shape[1], x_raw_full.shape[2]], dtype=np.float64)
for i in range(x_raw_full.shape[0]):
    for j in range(x_raw_full.shape[1]):
        for k in range(x_raw_full.shape[2]):
            x_raw_full_new[i, j, k] = x_raw_full[i, j, k]

np.save(os.path.join(args.processed_path, 'features.npy'), x_raw_full_new)
x_raw_full = np.load(os.path.join(args.processed_path, 'features.npy'))

from sklearn import preprocessing

x_full = preprocessing.scale(np.reshape(x_raw_full, [n_depths * n_steps, x_raw_full.shape[-1]]))
x_full = np.reshape(x_full, [n_depths, n_steps, x_full.shape[-1]])
np.save(os.path.join(args.processed_path, 'processed_features.npy'), x_full)

# label_glm
glm_new = glm.drop('date', axis=1).values
glm_new = np.transpose(glm_new)

labels = np.zeros([n_depths, n_steps], dtype=np.float64)
for i in range(n_depths):
    for j in range(n_steps):
        labels[i, j] = glm_new[i, j]

np.save(os.path.join(args.processed_path, 'labels_pretrain.npy'), labels)

# Ice mask (so we don't penalize energy imbalance on days with ice)
diag_all = pd.read_csv(args.ice_file)
diag_merged = diag_all.merge(feat, how='right', on='date')[['ice']].values

diag = np.zeros([n_depths, n_steps, 3], dtype=np.float64)
for i in range(n_depths):
    for j in range(n_steps):
        diag[i, j, 2] = diag_merged[j, 0]
np.save(os.path.join(args.processed_path, 'diag.npy'), diag)  # ['ignored', 'ignored', 'ice']

print("Processed data are in %s" % args.processed_path)
"""
