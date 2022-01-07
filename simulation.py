from init import init_agents_data, init_agents_pos


def simulate(world, mover, data_recorder, data_modifier, params):
    for rep in range(params.num_repeats):
        agents_pos = init_agents_pos(params)
        agents_data = init_agents_data(params)
        for step in range(params.num_steps):
            new_agents_pos, agents_data = mover.step(agents_pos, agents_data, params)
            new_agents_pos = world.step(agents_pos, new_agents_pos, params)
            data_recorder.record(agents_pos, new_agents_pos, agents_data, world, mover, rep, step)
            data_modifier.modify(agents_data, params)
            agents_pos = new_agents_pos
            
        data_recorder.new_repeat()
        data_modifier.new_repeat(agents_data, params)

    return data_recorder.get_results()
