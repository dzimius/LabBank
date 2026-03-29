report_date = None
balance_start_date = None
bs_side_map = {"E": "0","A": "1", "L": "2", "O": "3"}
rate_type_map = {"":"0", "F": "1", "V": "2", "A": "3"}
avg_rate_map = {"mortgage_fixed": 6,
                "mortgage_float": 5.8,
                "cash_loan_fixed": 8.5,
                "cash_loan_float": 8.9,
                "investment_loan_fixed": 5.5,
                "investment_loan_float": 5.9}

stats_dict = {
        "current_account": {
                2: {'mean': 300_000, 'std_dev': 250_000, 'lower_bound': 0, 'upper_bound': 30_000_000},
                4: {'mean': 45_000,  'std_dev': 100_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
                5: {'mean': 200_000, 'std_dev': 200_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
                6: {'mean': 60_000,  'std_dev': 80_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
                7: {'mean': 5_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
                8: {'mean': 7_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000}
                },
        "saving_account": {
                2: {'mean': 300_000, 'std_dev': 250_000, 'lower_bound': 0, 'upper_bound': 30_000_000},
                4: {'mean': 45_000,  'std_dev': 100_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
                5: {'mean': 200_000, 'std_dev': 200_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
                6: {'mean': 60_000,  'std_dev': 80_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
                7: {'mean': 5_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
                8: {'mean': 7_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000}
                },
        "term_deposit": {
                2: {'mean': 300_000, 'std_dev': 250_000, 'lower_bound': 0, 'upper_bound': 30_000_000},
                4: {'mean': 20_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
                5: {'mean': 200_000, 'std_dev': 150_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
                6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
                7: {'mean': 5_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
                8: {'mean': 7_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
                9: {'mean': 50_000_000, 'std_dev': 30_000_000, 'lower_bound': 5_000_000, 'upper_bound': 200_000_000},
                },
        "mortgage_fixed": {
                4: {'mean': 600_000,  'std_dev': 350_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
                5: {'mean': 200_000, 'std_dev': 150_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
                6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000}
                },
        "mortgage_float": {
                4: {'mean': 600_000,  'std_dev': 350_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
                5: {'mean': 200_000, 'std_dev': 150_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
                6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000}
                },
        "cash_loan_fixed": {
                4: {'mean': 60_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
                6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000}
                },
        "cash_loan_float": {
                4: {'mean': 60_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
                6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000}
                },
        "investment_loan_fixed": {
                5: {'mean': 1_000_000, 'std_dev': 2_000_000, 'lower_bound': 300_000, 'upper_bound': 10_000_000},
                7: {'mean': 15_000_000, 'std_dev': 15_000_000, 'lower_bound': 2_000_000, 'upper_bound': 70_000_000}
                },
        "investment_loan_float": {
                5: {'mean': 1_000_000, 'std_dev': 2_000_000, 'lower_bound': 300_000, 'upper_bound': 10_000_000},
                7: {'mean': 15_000_000, 'std_dev': 15_000_000, 'lower_bound': 2_000_000, 'upper_bound': 70_000_000}
                },
        "bond_fixed": {
                1: {'mean': 50_000_000,  'std_dev': 70_000_000,  'lower_bound': 5_000_000, 'upper_bound': 150_000_000},
                2: {'mean': 2_000_000, 'std_dev': 2_000_000, 'lower_bound': 100_000, 'upper_bound': 20_000_000},
                7: {'mean': 2_000_000,  'std_dev': 3_000_000,  'lower_bound': 0, 'upper_bound': 20_000_000},
                8: {'mean': 2_000_000,  'std_dev': 3_000_000,  'lower_bound': 0, 'upper_bound': 20_000_000}
                },
        "bond_float": {
                1: {'mean': 50_000_000, 'std_dev': 70_000_000, 'lower_bound': 5_000_000, 'upper_bound': 150_000_000},
                2: {'mean': 2_000_000, 'std_dev': 2_000_000, 'lower_bound': 100_000, 'upper_bound': 20_000_000},
                7: {'mean': 2_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 20_000_000},
                8: {'mean': 2_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 20_000_000}
                },
        "issued_bond": {
                8: {'mean': 50_000_000, 'std_dev': 40_000_000, 'lower_bound': 10_000_000, 'upper_bound': 200_000_000},
                }
}

margin_map = {
        "mortgage_fixed": {'mean': 2/100,  'std_dev': 0.5/100,  'lower_bound': 0.25/100, 'upper_bound': 4/100},
        "mortgage_float": {'mean': 2/100,  'std_dev': 0.5/100,  'lower_bound': 0.25/100, 'upper_bound': 4/100},
        "cash_loan_fixed": {'mean': 4.5/100,  'std_dev': 0.5/100,  'lower_bound': 3/100, 'upper_bound': 7/100},
        "cash_loan_float": {'mean': 4.5/100,  'std_dev': 0.5/100,  'lower_bound': 3/100, 'upper_bound': 7/100},
        "investment_loan_fixed": {'mean': 1/100, 'std_dev': 1/100, 'lower_bound': 0.5/100, 'upper_bound': 5/100},
        "investment_loan_float": {'mean': 1/100, 'std_dev': 1/100, 'lower_bound': 0.5/100, 'upper_bound': 5/100}
        }