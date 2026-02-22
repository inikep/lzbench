extern struct param_data {
  uint32_t max_rules;
  uint8_t cap_encoded, cap_lock_disabled, delta_disabled, create_words, fast_mode, user_set_RAM_size;
  uint8_t user_set_profit_ratio_power, user_set_production_cost, print_dictionary, use_mtf, two_threads;
  double RAM_usage, production_cost, order, profit_ratio_power;
} params;
