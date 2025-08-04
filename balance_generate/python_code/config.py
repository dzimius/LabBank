report_date = None
bs_side_map = {"E": "0","A": "1", "L": "2", "O": "3"}
rate_type_map = {"":"0", "F": "1", "V": "2", "A": "3"}
c_acc_stats_dict = {
        2: {'mean': 300_000, 'std_dev': 250_000, 'lower_bound': 0, 'upper_bound': 30_000_000},
        4: {'mean': 45_000,  'std_dev': 100_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        5: {'mean': 200_000, 'std_dev': 200_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
        6: {'mean': 60_000,  'std_dev': 80_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        7: {'mean': 5_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
        8: {'mean': 7_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000}
    }
s_acc_stats_dict = {
        2: {'mean': 300_000, 'std_dev': 250_000, 'lower_bound': 0, 'upper_bound': 30_000_000},
        4: {'mean': 45_000,  'std_dev': 100_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        5: {'mean': 200_000, 'std_dev': 200_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
        6: {'mean': 60_000,  'std_dev': 80_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        7: {'mean': 5_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
        8: {'mean': 7_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000}
}
t_dep_stats_dict = {
        2: {'mean': 300_000, 'std_dev': 250_000, 'lower_bound': 0, 'upper_bound': 30_000_000},
        4: {'mean': 20_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        5: {'mean': 200_000, 'std_dev': 150_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
        6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        7: {'mean': 5_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
        8: {'mean': 7_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000}
    }
mortg_stats_dict = {
        4: {'mean': 600_000,  'std_dev': 350_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        5: {'mean': 200_000, 'std_dev': 150_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
        6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000}
    }
c_loan_stats_dict = {
        4: {'mean': 100_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},

    }