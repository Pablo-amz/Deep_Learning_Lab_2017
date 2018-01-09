
import ConfigSpace as CS

cs = CS.ConfigurationSpace()


# Uniformly distributed discrete (Integer) hyperparameter

discrete_hp = CS.UniformIntegerHyperparameter("discrete_hp",
                                               lower=1,
                                               default_value=2,
                                               upper=10)
cs.add_hyperparameter(discrete_hp)

# Uniformly distributed float hyperparameter

float_hp = CS.UniformFloatHyperparameter('float_hp',
                                       lower=0,
                                       upper=1,
                                       default_value=0.5,
                                       log=False)

cs.add_hyperparameter(float_hp)

# Uniformly distributed float hyperparameter on a logarithmic scale

log_float_hp = CS.UniformFloatHyperparameter('log_float_hp',
                                       lower=1e-6,
                                       upper=1e-2,
                                       default_value=1e-4,
                                       log=True)

cs.add_hyperparameter(log_float_hp)

# Uniformly distributed categorical hyperparameter 

cat_hp = CS.CategoricalHyperparameter("cat_hp", ['a', 'b'])
cs.add_hyperparameter(cat_hp)

# Add condition that the hyperparameter log_float_hp is only active if cat_hp is set to 'a'

cond = CS.EqualsCondition(log_float_hp, cat_hp, 'a')
cs.add_condition(cond)

# Add condition that the hyperparameter float_hp is only active if discrete_hp has an higher or equal value than 2

cond = CS.GreaterThanCondition(float_hp, discrete_hp, 2)
cs.add_condition(cond)

print(cs.sample_configuration())
