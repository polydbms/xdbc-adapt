########
# Environments to test scaling the compute power
#######
environment_101 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 20,
    "timeout": 550
}

environment_102 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 50,
    "timeout": 550
}

environment_103 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 100,
    "timeout": 550
}

environment_104 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 150,
    "timeout": 550
}

environment_105 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 200,
    "timeout": 550
}

environment_106 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 300,
    "timeout": 550
}

environment_107 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 400,
    "timeout": 550
}

environment_108 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 500,
    "timeout": 550
}

environment_109 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 600,
    "timeout": 550
}

environment_110 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 700,
    "timeout": 550
}

environment_111 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 800,
    "timeout": 550
}

environment_112 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 900,
    "timeout": 550
}

environment_113 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 550
}

#all environments with cpu == 16
envs_scale_network_test = [environment_101, environment_102, environment_103,
                           environment_104, environment_105, environment_106,
                           environment_107, environment_108, environment_109,
                           environment_110, environment_111, environment_112,
                           environment_113]

########
# Environments to test scaling the network speed
#######

environment_301 = {
    "server_cpu": 2,
    "client_cpu": 2,
    "network": 1000,
    "timeout": 550
}

environment_302 = {
    "server_cpu": 2,
    "client_cpu": 4,
    "network": 1000,
    "timeout": 550
}

environment_303 = {
    "server_cpu": 2,
    "client_cpu": 8,
    "network": 1000,
    "timeout": 550
}

environment_304 = {
    "server_cpu": 2,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 550
}

environment_305 = {
    "server_cpu": 4,
    "client_cpu": 2,
    "network": 1000,
    "timeout": 550
}

environment_306 = {
    "server_cpu": 4,
    "client_cpu": 4,
    "network": 1000,
    "timeout": 550
}

environment_307 = {
    "server_cpu": 4,
    "client_cpu": 8,
    "network": 1000,
    "timeout": 550
}

environment_308 = {
    "server_cpu": 4,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 550
}

environment_309 = {
    "server_cpu": 8,
    "client_cpu": 2,
    "network": 1000,
    "timeout": 550
}

environment_310 = {
    "server_cpu": 8,
    "client_cpu": 4,
    "network": 1000,
    "timeout": 550
}

environment_311 = {
    "server_cpu": 8,
    "client_cpu": 8,
    "network": 1000,
    "timeout": 550
}

environment_312 = {
    "server_cpu": 8,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 550
}

environment_313 = {
    "server_cpu": 16,
    "client_cpu": 2,
    "network": 1000,
    "timeout": 550
}

environment_314 = {
    "server_cpu": 16,
    "client_cpu": 4,
    "network": 1000,
    "timeout": 550
}

environment_315 = {
    "server_cpu": 16,
    "client_cpu": 8,
    "network": 1000,
    "timeout": 550
}

environment_316 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 550
}

envs_scale_compute_test = [environment_301, environment_302, environment_303,
                           environment_304, environment_305, environment_306,
                           environment_307, environment_308, environment_309,
                           environment_310, environment_311, environment_312,
                           environment_313, environment_314, environment_315,
                           environment_316]

# final environments selection
'''
environment_1 = {
    "server_cpu": 2,
    "client_cpu": 2,
    "network": 50,
    "timeout": 500
}
environment_2 = {
    "server_cpu": 2,
    "client_cpu": 8,
    "network": 50,
    "timeout": 450
}
environment_3 = {
    "server_cpu": 4,
    "client_cpu": 16,
    "network": 50,
    "timeout": 450
}
environment_4 = {
    "server_cpu": 8,
    "client_cpu": 8,
    "network": 50,
    "timeout": 450
}
environment_5 = {
    "server_cpu": 8,
    "client_cpu": 2,
    "network": 50,
    "timeout": 500
}
environment_6 = {
    "server_cpu": 8,
    "client_cpu": 16,
    "network": 50,
    "timeout": 400
}
environment_7 = {
    "server_cpu": 16,
    "client_cpu": 4,
    "network": 50,
    "timeout": 400
}
environment_8 = {
    "server_cpu": 16,
    "client_cpu": 8,
    "network": 50,
    "timeout": 400
}
environment_9 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 50,
    "timeout": 400
}


environment_10 = {
    "server_cpu": 2,
    "client_cpu": 2,
    "network": 150,
    "timeout": 500
}
environment_11 = {
    "server_cpu": 2,
    "client_cpu": 8,
    "network": 150,
    "timeout": 450
}
environment_12 = {
    "server_cpu": 4,
    "client_cpu": 16,
    "network": 150,
    "timeout": 300
}
environment_13 = {
    "server_cpu": 8,
    "client_cpu": 8,
    "network": 150,
    "timeout": 300
}
environment_14 = {
    "server_cpu": 8,
    "client_cpu": 2,
    "network": 150,
    "timeout": 500
}
environment_15 = {
    "server_cpu": 8,
    "client_cpu": 16,
    "network": 150,
    "timeout": 300
}
environment_16 = {
    "server_cpu": 16,
    "client_cpu": 4,
    "network": 150,
    "timeout": 300
}
environment_17 = {
    "server_cpu": 16,
    "client_cpu": 8,
    "network": 150,
    "timeout": 300
}
environment_18 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 150,
    "timeout": 300
}


environment_19 = {
    "server_cpu": 2,
    "client_cpu": 2,
    "network": 1000,
    "timeout": 500
}
environment_20 = {
    "server_cpu": 2,
    "client_cpu": 8,
    "network": 1000,
    "timeout": 450
}
environment_21 = {
    "server_cpu": 4,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 300
}
environment_22 = {
    "server_cpu": 8,
    "client_cpu": 8,
    "network": 1000,
    "timeout": 300
}
environment_23 = {
    "server_cpu": 8,
    "client_cpu": 2,
    "network": 1000,
    "timeout": 500
}
environment_24 = {
    "server_cpu": 8,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 300
}
environment_25 = {
    "server_cpu": 16,
    "client_cpu": 4,
    "network": 1000,
    "timeout": 300
}
environment_26 = {
    "server_cpu": 16,
    "client_cpu": 8,
    "network": 1000,
    "timeout": 300
}
environment_27 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 300
}

environment_28 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 100000,
    "timeout": 300
}



environment_31 = {
    "server_cpu": 2,
    "client_cpu": 16,
    "network": 50,
    "timeout": 500
}
environment_32 = {
    "server_cpu": 2,
    "client_cpu": 16,
    "network": 150,
    "timeout": 500
}

environment_33 = {
    "server_cpu": 2,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 500
}

environment_34 = {
    "server_cpu": 16,
    "client_cpu": 2,
    "network": 50,
    "timeout": 500
}
environment_35 = {
    "server_cpu": 16,
    "client_cpu": 2,
    "network": 150,
    "timeout": 500
}

environment_36 = {
    "server_cpu": 16,
    "client_cpu": 2,
    "network": 1000,
    "timeout": 500
}

env_S2_C16_N50 = environment_31
env_S2_C16_N150 = environment_32
env_S2_C16_N1000 = environment_33

env_S16_C2_N50 = environment_34
env_S16_C2_N150 = environment_35
env_S16_C2_N1000 = environment_36


env_S2_C2_N50 = environment_1
env_S2_C8_N50 = environment_2
env_S4_C16_N50 = environment_3
env_S8_C8_N50 = environment_4
env_S8_C2_N50 = environment_5
env_S8_C16_N50 = environment_6
env_S16_C4_N50 = environment_7
env_S16_C8_N50 = environment_8
env_S16_C16_N50 = environment_9

env_S2_C2_N150 = environment_10
env_S2_C8_N150 = environment_11
env_S4_C16_N150 = environment_12
env_S8_C8_N150 = environment_13
env_S8_C2_N150 = environment_14
env_S8_C16_N150 = environment_15
env_S16_C4_N150 = environment_16
env_S16_C8_N150 = environment_17
env_S16_C16_N150 = environment_18

env_S2_C2_N1000 = environment_19
env_S2_C8_N1000 = environment_20
env_S4_C16_N1000 = environment_21
env_S8_C8_N1000 = environment_22
env_S8_C2_N1000 = environment_23
env_S8_C16_N1000 = environment_24
env_S16_C4_N1000 = environment_25
env_S16_C8_N1000 = environment_26
env_S16_C16_N1000 = environment_27

env_S16_C16_100000 = environment_28

all_final_base_signatures = ["S2_C2_N50","S2_C8_N50","S4_C16_N50","S8_C8_N50","S8_C2_N50","S8_C16_N50","S16_C4_N50","S16_C8_N50","S16_C16_N50",
                             "S2_C2_N150","S2_C8_N150","S4_C16_N150","S8_C8_N150","S8_C2_N150","S8_C16_N150","S16_C4_N150","S16_C8_N150","S16_C16_N150",
                             "S2_C2_N1000","S2_C8_N1000","S4_C16_N1000","S8_C8_N1000","S8_C2_N1000","S8_C16_N1000","S16_C4_N1000","S16_C8_N1000","S16_C16_N1000"]

all_final_main_signatures = ["S2_C2_N50","S2_C8_N50","S4_C16_N50",
                             "S8_C8_N150","S8_C2_N150","S8_C16_N150",
                             "S16_C4_N1000","S16_C8_N1000","S16_C16_N1000"]


environment_list_base_envs = [env_S2_C2_N50,env_S2_C8_N50,env_S4_C16_N50,
                                         env_S8_C8_N50,env_S8_C2_N50,env_S8_C16_N50,
                                         env_S16_C4_N50,env_S16_C8_N50,env_S16_C16_N50,

                                         env_S2_C2_N150,env_S2_C8_N150,env_S4_C16_N150,
                                         env_S8_C8_N150,env_S8_C2_N150,env_S8_C16_N150,
                                         env_S16_C4_N150,env_S16_C8_N150,env_S16_C16_N150,

                                         env_S2_C2_N1000,env_S2_C8_N1000,env_S4_C16_N1000,
                                         env_S8_C8_N1000,env_S8_C2_N1000,env_S8_C16_N1000,
                                         env_S16_C4_N1000,env_S16_C8_N1000,env_S16_C16_N1000]

environment_list_main_envs = [env_S2_C2_N50,
                              env_S2_C8_N50,
                              env_S4_C16_N50,

                              env_S8_C8_N150,
                              env_S8_C2_N150,
                              env_S8_C16_N150,

                              env_S16_C4_N1000,
                              env_S16_C8_N1000,
                              env_S16_C16_N1000]





environment_list_main_envs_3_times_sorted_slowest_first = [
    env_S2_C2_N50, env_S2_C2_N50, env_S2_C2_N50,
    env_S8_C2_N150, env_S8_C2_N150, env_S8_C2_N150,
    env_S2_C8_N50, env_S2_C8_N50, env_S2_C8_N50,

    env_S4_C16_N50, env_S4_C16_N50, env_S4_C16_N50,
    env_S16_C4_N1000, env_S16_C4_N1000, env_S16_C4_N1000,
    env_S8_C8_N150, env_S8_C8_N150, env_S8_C8_N150,

    env_S8_C16_N150, env_S8_C16_N150, env_S8_C16_N150,
    env_S16_C8_N1000, env_S16_C8_N1000, env_S16_C8_N1000,
    env_S16_C16_N1000, env_S16_C16_N1000, env_S16_C16_N1000
]

environment_list_main_envs_2_times_sorted_slowest_first = [
    env_S2_C2_N50, env_S2_C2_N50,
    env_S8_C2_N150, env_S8_C2_N150,
    env_S2_C8_N50, env_S2_C8_N50,

    env_S4_C16_N50, env_S4_C16_N50,
    env_S16_C4_N1000, env_S16_C4_N1000,
    env_S8_C8_N150, env_S8_C8_N150,

    env_S8_C16_N150, env_S8_C16_N150,
    env_S16_C8_N1000, env_S16_C8_N1000,
    env_S16_C16_N1000, env_S16_C16_N1000
]
'''

environment_1 = {
    "server_cpu": 2,
    "client_cpu": 2,
    "network": 50,
    "timeout": 750
}

environment_2 = {
    "server_cpu": 2,
    "client_cpu": 2,
    "network": 150,
    "timeout": 750
}

environment_3 = {
    "server_cpu": 2,
    "client_cpu": 2,
    "network": 1000,
    "timeout": 750
}

environment_4 = {
    "server_cpu": 2,
    "client_cpu": 2,
    "network": 100000,
    "timeout": 750
}

environment_5 = {
    "server_cpu": 2,
    "client_cpu": 8,
    "network": 50,
    "timeout": 750
}

environment_6 = {
    "server_cpu": 2,
    "client_cpu": 8,
    "network": 150,
    "timeout": 750
}

environment_7 = {
    "server_cpu": 2,
    "client_cpu": 8,
    "network": 1000,
    "timeout": 750
}

environment_8 = {
    "server_cpu": 2,
    "client_cpu": 8,
    "network": 100000,
    "timeout": 750
}

environment_9 = {
    "server_cpu": 2,
    "client_cpu": 16,
    "network": 50,
    "timeout": 750
}

environment_10 = {
    "server_cpu": 2,
    "client_cpu": 16,
    "network": 150,
    "timeout": 750
}

environment_11 = {
    "server_cpu": 2,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 750
}

environment_12 = {
    "server_cpu": 2,
    "client_cpu": 16,
    "network": 100000,
    "timeout": 750
}

environment_13 = {
    "server_cpu": 8,
    "client_cpu": 2,
    "network": 50,
    "timeout": 1050
}

environment_14 = {
    "server_cpu": 8,
    "client_cpu": 2,
    "network": 150,
    "timeout": 750
}

environment_15 = {
    "server_cpu": 8,
    "client_cpu": 2,
    "network": 1000,
    "timeout": 750
}

environment_16 = {
    "server_cpu": 8,
    "client_cpu": 2,
    "network": 100000,
    "timeout": 750
}

environment_17 = {
    "server_cpu": 8,
    "client_cpu": 8,
    "network": 50,
    "timeout": 550
}

environment_18 = {
    "server_cpu": 8,
    "client_cpu": 8,
    "network": 150,
    "timeout": 550
}

environment_19 = {
    "server_cpu": 8,
    "client_cpu": 8,
    "network": 1000,
    "timeout": 550
}

environment_20 = {
    "server_cpu": 8,
    "client_cpu": 8,
    "network": 100000,
    "timeout": 550
}

environment_21 = {
    "server_cpu": 8,
    "client_cpu": 16,
    "network": 50,
    "timeout": 550
}

environment_22 = {
    "server_cpu": 8,
    "client_cpu": 16,
    "network": 150,
    "timeout": 550
}

environment_23 = {
    "server_cpu": 8,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 550
}

environment_24 = {
    "server_cpu": 8,
    "client_cpu": 16,
    "network": 100000,
    "timeout": 550
}

environment_25 = {
    "server_cpu": 16,
    "client_cpu": 2,
    "network": 50,
    "timeout": 750
}

environment_26 = {
    "server_cpu": 16,
    "client_cpu": 2,
    "network": 150,
    "timeout": 750
}

environment_27 = {
    "server_cpu": 16,
    "client_cpu": 2,
    "network": 1000,
    "timeout": 750
}

environment_28 = {
    "server_cpu": 16,
    "client_cpu": 2,
    "network": 100000,
    "timeout": 750
}

environment_29 = {
    "server_cpu": 16,
    "client_cpu": 8,
    "network": 50,
    "timeout": 550
}

environment_30 = {
    "server_cpu": 16,
    "client_cpu": 8,
    "network": 150,
    "timeout": 550
}

environment_31 = {
    "server_cpu": 16,
    "client_cpu": 8,
    "network": 1000,
    "timeout": 550
}

environment_32 = {
    "server_cpu": 16,
    "client_cpu": 8,
    "network": 100000,
    "timeout": 550
}

environment_33 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 50,
    "timeout": 550
}

environment_34 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 150,
    "timeout": 550
}

environment_35 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 1000,
    "timeout": 550
}

environment_36 = {
    "server_cpu": 16,
    "client_cpu": 16,
    "network": 100000,
    "timeout": 550
}

env_S2_C2_N50 = environment_1
env_S2_C2_N150 = environment_2
env_S2_C2_N1000 = environment_3
env_S2_C2_N100000 = environment_4
env_S2_C8_N50 = environment_5
env_S2_C8_N150 = environment_6
env_S2_C8_N1000 = environment_7
env_S2_C8_N100000 = environment_8
env_S2_C16_N50 = environment_9
env_S2_C16_N150 = environment_10
env_S2_C16_N1000 = environment_11
env_S2_C16_N100000 = environment_12
env_S8_C2_N50 = environment_13
env_S8_C2_N150 = environment_14
env_S8_C2_N1000 = environment_15
env_S8_C2_N100000 = environment_16
env_S8_C8_N50 = environment_17
env_S8_C8_N150 = environment_18
env_S8_C8_N1000 = environment_19
env_S8_C8_N100000 = environment_20
env_S8_C16_N50 = environment_21
env_S8_C16_N150 = environment_22
env_S8_C16_N1000 = environment_23
env_S8_C16_N100000 = environment_24
env_S16_C2_N50 = environment_25
env_S16_C2_N150 = environment_26
env_S16_C2_N1000 = environment_27
env_S16_C2_N100000 = environment_28
env_S16_C8_N50 = environment_29
env_S16_C8_N150 = environment_30
env_S16_C8_N1000 = environment_31
env_S16_C8_N100000 = environment_32
env_S16_C16_N50 = environment_33
env_S16_C16_N150 = environment_34
env_S16_C16_N1000 = environment_35
env_S16_C16_N100000 = environment_36

all_base_signatures = [
    "S2_C2_N50", "S2_C2_N150", "S2_C2_N1000", "S2_C2_N100000",
    "S2_C8_N50", "S2_C8_N150", "S2_C8_N1000", "S2_C8_N100000",
    "S2_C16_N50", "S2_C16_N150", "S2_C16_N1000", "S2_C16_N100000",
    "S8_C2_N50", "S8_C2_N150", "S8_C2_N1000", "S8_C2_N100000",
    "S8_C8_N50", "S8_C8_N150", "S8_C8_N1000", "S8_C8_N100000",
    "S8_C16_N50", "S8_C16_N150", "S8_C16_N1000", "S8_C16_N100000",
    "S16_C2_N50", "S16_C2_N150", "S16_C2_N1000", "S16_C2_N100000",
    "S16_C8_N50", "S16_C8_N150", "S16_C8_N1000", "S16_C8_N100000",
    "S16_C16_N50", "S16_C16_N150", "S16_C16_N1000", "S16_C16_N100000",
]

all_main_signatures = [
    "S2_C2_N50",
    "S2_C8_N50",
    "S2_C16_N50",

    "S8_C2_N150",
    "S8_C8_N150",
    "S8_C16_N150",

    "S16_C2_N1000",
    "S16_C8_N1000",
    "S16_C16_N1000",
]

all_base_environments = [
    env_S2_C2_N50, env_S2_C2_N150, env_S2_C2_N1000, env_S2_C2_N100000,
    env_S2_C8_N50, env_S2_C8_N150, env_S2_C8_N1000, env_S2_C8_N100000,
    env_S2_C16_N50, env_S2_C16_N150, env_S2_C16_N1000, env_S2_C16_N100000,
    env_S8_C2_N50, env_S8_C2_N150, env_S8_C2_N1000, env_S8_C2_N100000,
    env_S8_C8_N50, env_S8_C8_N150, env_S8_C8_N1000, env_S8_C8_N100000,
    env_S8_C16_N50, env_S8_C16_N150, env_S8_C16_N1000, env_S8_C16_N100000,
    env_S16_C2_N50, env_S16_C2_N150, env_S16_C2_N1000, env_S16_C2_N100000,
    env_S16_C8_N50, env_S16_C8_N150, env_S16_C8_N1000, env_S16_C8_N100000,
    env_S16_C16_N50, env_S16_C16_N150, env_S16_C16_N1000, env_S16_C16_N100000
]

all_main_environments = [
    env_S2_C2_N50,
    env_S2_C8_N50,
    env_S2_C16_N50,

    env_S8_C2_N150,
    env_S8_C8_N150,
    env_S8_C16_N150,

    env_S16_C2_N1000,
    env_S16_C8_N1000,
    env_S16_C16_N1000,
]


all_main_environments_4_times_sorted_slowest_first = [
    env_S2_C2_N50,env_S2_C2_N50,env_S2_C2_N50,env_S2_C2_N50,
    env_S8_C2_N150,env_S8_C2_N150,env_S8_C2_N150,env_S8_C2_N150,
    env_S16_C2_N1000,env_S16_C2_N1000,env_S16_C2_N1000,env_S16_C2_N1000,
    env_S2_C8_N50,env_S2_C8_N50,env_S2_C8_N50,env_S2_C8_N50,
    env_S2_C16_N50,env_S2_C16_N50,env_S2_C16_N50,env_S2_C16_N50,
    env_S8_C8_N150,env_S8_C8_N150,env_S8_C8_N150,env_S8_C8_N150,
    env_S8_C16_N150,env_S8_C16_N150,env_S8_C16_N150,env_S8_C16_N150,
    env_S16_C8_N1000,env_S16_C8_N1000,env_S16_C8_N1000,env_S16_C8_N1000,
    env_S16_C16_N1000,env_S16_C16_N1000,env_S16_C16_N1000,env_S16_C16_N1000,
]


all_main_environments_3_times_sorted_slowest_first = [
    env_S2_C2_N50,env_S2_C2_N50,env_S2_C2_N50,
    env_S8_C2_N150,env_S8_C2_N150,env_S8_C2_N150,
    env_S16_C2_N1000,env_S16_C2_N1000,env_S16_C2_N1000,
    env_S2_C8_N50,env_S2_C8_N50,env_S2_C8_N50,
    env_S2_C16_N50,env_S2_C16_N50,env_S2_C16_N50,
    env_S8_C8_N150,env_S8_C8_N150,env_S8_C8_N150,
    env_S8_C16_N150,env_S8_C16_N150,env_S8_C16_N150,
    env_S16_C8_N1000,env_S16_C8_N1000,env_S16_C8_N1000,
    env_S16_C16_N1000,env_S16_C16_N1000,env_S16_C16_N1000,
]

all_main_environments_2_times_sorted_slowest_first = [
    env_S2_C2_N50,env_S2_C2_N50,
    env_S8_C2_N150,env_S8_C2_N150,
    env_S16_C2_N1000,env_S16_C2_N1000,
    env_S2_C8_N50,env_S2_C8_N50,
    env_S2_C16_N50,env_S2_C16_N50,
    env_S8_C8_N150,env_S8_C8_N150,
    env_S8_C16_N150,env_S8_C16_N150,
    env_S16_C8_N1000,env_S16_C8_N1000,
    env_S16_C16_N1000,env_S16_C16_N1000,
]
