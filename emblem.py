# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:17:01 2024

@author: bnguyen2
"""

import matplotlib.pyplot as plt
import numba
import numpy as np


@numba.njit
def damage_multiplier(attack=2, 
                      boss_attack=1, boss_attack_hypers=np.array([0]),
                      critical_damage=3.5, critical_damage_hypers=np.array([0]),
                      damage=0, damage_nodes=np.array([0.1]),
                      final_damage=0.8, final_damage_hypers=np.array([0.2]), final_damage_nodes=np.array([1.2]), 
                      hit_count=np.array([1]), 
                      ied_boss=0, ied_nodes=np.array([0]), ied_sources=np.array([0]),
                      skill_damage=np.array([2.2])):
    """
    This function computes the average damage multiplier excluding raw attack.
    hit count * (1 + 0.2 + critical damage) * 
    (1 + attack + skill damage * boss attack) *
    (1 + damage) * (1 + final damage)

    Parameters
    ----------
    attack : TYPE, float
        Physical or magical attack. The default is 2 (200%).
    boss_attack : TYPE, float
        Boss attack. The default is 1 (100%).
    boss_attack_hypers : TYPE, numpy array of floats
        Boss attack from hyper skills. The default is np.array([0]).
    critical_damage : TYPE, float
        Critical damage. The default is 3.5 (350%).
    critical_damage_hypers : TYPE, numpy array of floats
        Critical damage from hyper skills. The default is np.array([0]).
    damage : TYPE, float
        Physical or magical damage. The default is 0 (0%).
    damage_nodes : TYPE, numpy array of floats
        Physical or magical damage. The default is np.array([0]).
    final_damage : TYPE, float
        Final damage. The default is 0.8 (80%).
    final_damage_hypers : TYPE, numpy array of floats
        Final damage from hyper skills. The default is np.array([0.2]).
    final_damage_nodes : TYPE, numpy array of floats
        Final damage from nodes. The default is np.array([1.2]).
    ied_boss : TYPE, float
        Ignore enemy defense of boss. The default is 0 (0%).
    ied_nodes : TYPE, numpy array of floats
        Ignore enemy defense from nodes. The default is np.array([0]).
    ied_sources : TYPE, numpy array of floats
        Ignore enemy defense from other sources such as equipment, except nodes. 
        The default is np.array([0]).
    hit_count : TYPE, numpy array of integers
        Hit count based on a certain amount of time. The default is np.array([1]).
    skill_damage : TYPE, numpy array of floats
        Skill damage. The default is np.array([2.2]).

    Returns
    -------
    TYPE, float
        Damage multiplier
    """
    result = 0
    
    total_boss_attack = boss_attack + boss_attack_hypers
    total_critical_damage = critical_damage + critical_damage_hypers
    total_damage = damage + damage_nodes
    total_final_damage = final_damage + final_damage_hypers + final_damage_nodes
    
    total_ied = ied_boss*(1 - ied_nodes)
    for ied in ied_sources:
        total_ied *= 1 - ied
    total_ied = 1 - total_ied
    
    
    for hc, sd, ba, cd, d, fd, ied in zip(hit_count, skill_damage, \
                                          total_boss_attack, \
                                          total_critical_damage, \
                                          total_damage, \
                                          total_final_damage, \
                                          total_ied):
        result += hc * (1.2+cd) * (1+attack+sd*ba) * (1 + d) * (1 + fd)
        if ied_boss > 0:
            result *= ied
        
    return result / sum(hit_count)


@numba.njit
def damage_data_emblem(attack=2, 
                       boss_attack_hypers=np.array([0]),
                       critical_damage_hypers=np.array([0]),
                       damage=0, damage_nodes=np.array([0.1]),
                       final_damage=0.8, final_damage_hypers=np.array([0.2]), final_damage_nodes=np.array([1.2]),
                       h=0.001,
                       hit_count=np.array([1]), 
                       ied_boss=0, ied_nodes=np.array([0]), ied_sources=np.array([0]),
                       max_modifier=5,
                       skill_damage=np.array([2.2])):
    """
    Computes the damage multiplier of all 9 combinations of boss attack and 
    critical damage emblems (2 different emblems and 8 equipment slots). It is 
    assumed that the emblems are maxed. Thus boss attack emblem adds 10% boss
    attack and critical damage emblem adds 20% critical damage. For each emblem
    combination the damage multiplier is computed for a range of base boss attack
    and base critical damage. The range for both damage sources is [0, max_modifier].

    Parameters
    ----------
    attack : TYPE, float
        Physical or magical attack. The default is 2 (200%).
    boss_attack_hypers : TYPE, numpy array of floats
        Boss attack from hyper skills. The default is np.array([0]).
    critical_damage_hypers : TYPE, numpy array of floats
        Critical damage from hyper skills. The default is np.array([0]).
    damage : TYPE, float
        Physical or magical damage. The default is 0 (0%).
    damage_nodes : TYPE, numpy array of floats
        Physical or magical damage. The default is np.array([0]).
    final_damage : TYPE, float
        Final damage. The default is 0.8 (80%).
    final_damage_hypers : TYPE, numpy array of floats
        Final damage from hyper skills. The default is np.array([0.2]).
    final_damage_nodes : TYPE, numpy array of floats
        Final damage from nodes. The default is np.array([1.2]).
    ied_boss : TYPE, float
        Ignore enemy defense of boss. The default is 0 (0%).
    ied_nodes : TYPE, numpy array of floats
        Ignore enemy defense from nodes. The default is np.array([0]).
    ied_sources : TYPE, numpy array of floats
        Ignore enemy defense from other sources such as equipment, except nodes. 
        The default is np.array([0]).
    h : TYPE, float
        Step size for boss attack and critical damage. The default is 0.001 (0.1%).
    hit_count : TYPE, numpy array of integers
        Hit count based on a certain amount of time. The default is np.array([1]).
    max_modifier : TYPE, integer
        Maximum boss attack and critical damage excluding emblem stats for each scenario. 
    skill_damage : TYPE, numpy array of floats
        Skill damage. The default is np.array([2.2]).

    Returns
    -------
    TYPE, list of numpy array
    Contains the damage multipliers for each emblem combination from each scenario.

    """
    n = int(max_modifier/h)+1

    boss_attack = np.array([i*h for i in range(n)])
    critical_damage = np.array([i*h for i in range(n)])

    damage_data = []

    for number_of_ba_emblem in range(9):
        damage_data_scenario = np.zeros((n, n))
        for i, ba in enumerate(boss_attack):
            for j, cd in enumerate(critical_damage):
                damage_data_scenario[i, j] = damage_multiplier(attack=attack,
                                                               boss_attack=ba + number_of_ba_emblem*0.1,
                                                               boss_attack_hypers = boss_attack_hypers,
                                                               critical_damage=cd +
                                                               (8-number_of_ba_emblem)*0.2,
                                                               critical_damage_hypers = critical_damage_hypers,
                                                               damage = damage, 
                                                               damage_nodes = damage_nodes,
                                                               final_damage = final_damage, 
                                                               final_damage_hypers = final_damage_hypers, 
                                                               final_damage_nodes = final_damage_nodes,
                                                               hit_count = hit_count,
                                                               ied_boss = ied_boss, 
                                                               ied_nodes = ied_nodes, 
                                                               ied_sources = ied_sources,
                                                               skill_damage=skill_damage)
        damage_data.append(damage_data_scenario)

    return damage_data

@numba.njit
def damage_data_emblem_one_scenario(attack=2,
                                    boss_attack = 1,
                                    boss_attack_hypers=np.array([0]),
                                    critical_damage = 3.5,
                                    critical_damage_hypers=np.array([0]),
                                    damage=0, damage_nodes=np.array([0.1]),
                                    final_damage=0.8, final_damage_hypers=np.array([0.2]), final_damage_nodes=np.array([1.2]),
                                    hit_count=np.array([1]), 
                                    ied_boss=0, ied_nodes=np.array([0]), ied_sources=np.array([0]),
                                    skill_damage=np.array([2.2])):
    """
    Computes the damage multiplier of all 9 combinations of boss attack and 
    critical damage emblems (2 different emblems and 8 equipment slots). It is 
    assumed that the emblems are maxed. Thus boss attack emblem adds 10% boss
    attack and critical damage emblem adds 20% critical damage. For each emblem
    combination the damage multiplier is computed for a specified boss attack
    and critical damage.

    Parameters
    ----------
    attack : TYPE, float
        Physical or magical attack. The default is 2 (200%).
    boss_attack : TYPE, float
        Boss attack. The default is 1 (100%).
    boss_attack_hypers : TYPE, numpy array of floats
        Boss attack from hyper skills. The default is np.array([0]).
    critical_damage : TYPE, float
        Critical damage. The default is 3.5 (350%).
    critical_damage_hypers : TYPE, numpy array of floats
        Critical damage from hyper skills. The default is np.array([0]).
    damage : TYPE, float
        Physical or magical damage. The default is 0 (0%).
    damage_nodes : TYPE, numpy array of floats
        Physical or magical damage. The default is np.array([0]).
    final_damage : TYPE, float
        Final damage. The default is 0.8 (80%).
    final_damage_hypers : TYPE, numpy array of floats
        Final damage from hyper skills. The default is np.array([0.2]).
    final_damage_nodes : TYPE, numpy array of floats
        Final damage from nodes. The default is np.array([1.2]).
    ied_boss : TYPE, float
        Ignore enemy defense of boss. The default is 0 (0%).
    ied_nodes : TYPE, numpy array of floats
        Ignore enemy defense from nodes. The default is np.array([0]).
    ied_sources : TYPE, numpy array of floats
        Ignore enemy defense from other sources such as equipment, except nodes. 
        The default is np.array([0]).
    hit_count : TYPE, numpy array of integers
        Hit count based on a certain amount of time. The default is np.array([1]).
    skill_damage : TYPE, numpy array of floats
        Skill damage. The default is np.array([2.2]).

    Returns
    -------
    TYPE, list of numpy array
    Contains the damage multipliers for each emblem combination from each scenario.

    """
    damage_data = np.zeros(9)

    for number_of_ba_emblem in range(9):
        damage_data[number_of_ba_emblem] = damage_multiplier(attack=attack,
                                                             boss_attack=boss_attack + number_of_ba_emblem*0.1,
                                                             boss_attack_hypers = boss_attack_hypers,
                                                             critical_damage=critical_damage + (8-number_of_ba_emblem)*0.2,
                                                             critical_damage_hypers = critical_damage_hypers,
                                                             damage = damage, 
                                                             damage_nodes = damage_nodes,
                                                             final_damage = final_damage, 
                                                             final_damage_hypers = final_damage_hypers, 
                                                             final_damage_nodes = final_damage_nodes,
                                                             hit_count = hit_count,
                                                             ied_boss = ied_boss, 
                                                             ied_nodes = ied_nodes, 
                                                             ied_sources = ied_sources,
                                                             skill_damage=skill_damage)
    
    emblem_ranking = damage_data.argsort()
    damage_data = damage_data[emblem_ranking]
    damage_data_ratio = damage_data / np.max(damage_data)
    
    return damage_data, damage_data_ratio, emblem_ranking


def damage_emblem_comparison(damage_data):
    """
    Searches for best emblem setting in the damage data from the ouput of the
    function damage_emblem_data.

    Parameters
    ----------
    damage_data : TYPE, list of numpy array
        Each numpy array should correspond to an emblem setting. Moreover, each
        numpy array has elements corresponding to a scenario, i.e. a given base
        boss attack and base critical damage.

    Returns
    -------
    damage_data_label : numpy array
        For each scenario the optimal emblem combination is given.
    damage_data_optimal : numpy array
        For each scenario the optimal damage multiplier is given.

    """
    damage_data_optimal = damage_data[0]
    damage_data_label = np.zeros(damage_data_optimal.shape, dtype=int)

    for i in range(1, len(damage_data)):
        damage_data_optimal_indices = damage_data_optimal < damage_data[i]
        damage_data_optimal[damage_data_optimal_indices] = damage_data[i][damage_data_optimal_indices]
        damage_data_label[damage_data_optimal_indices] = i

    return damage_data_label, damage_data_optimal


if __name__ == '__main__':
    # h = 0.01
    # max_modifier = 5

    # damage_data = damage_data_emblem(h=h, max_modifier=max_modifier)
    # damage_data_label, damage_data_optimal = damage_emblem_comparison(
    #     damage_data)

    # plt.figure(figsize=(8, 8), dpi=1000)
    # plt.imshow(damage_data_label,
    #             extent=[0, max_modifier*100, max_modifier*100, 0],
    #             interpolation='none')
    # plt.title('Optimal emblem combination based on BA and CD')
    # plt.xlabel("Critical Damage (%)")
    # plt.ylabel("Boss Attack (%)")
    # plt.colorbar(orientation='horizontal')
    
    damage_data, damage_data_ratio, emblem_ranking = damage_data_emblem_one_scenario(boss_attack=2.25,
                                                                                     critical_damage=3.5)
    
    for dd, ddr, er in zip(damage_data, damage_data_ratio, emblem_ranking):
        print("({} BA, {} CD), damage multiplier = {:.2f} and {:.2f}% worse than best".format(er, 8-er, dd, 100*(1-ddr)))