"""Input and uniform accessors for reverse shader expressions."""

def _reverse_uniform_components(contract, nodes):
    components = []
    for argument in contract.parameters:
        matches = _matching_uniform_nodes(nodes, argument.name)
        if argument.shape == ():
            components.append("+".join(f"g{i}" for i in matches) or "0.0")
        elif argument.shape == (2,):
            components.extend(
                "+".join(f"g{i}.{axis}" for i in matches) or "0.0" for axis in "xy"
            )
        else:
            raise ValueError("Reverse expression supports scalar and vec2 uniforms")
    return components


def _matching_uniform_nodes(nodes, name):
    return [
        index
        for index, (node, _, _, _) in enumerate(nodes)
        if node.op == "uniform" and node.args[0] == name
    ]


def _reverse_input_declarations(contract):
    return "\n".join(
        f"layout(r32f,binding={index}) readonly uniform image2D source{index};"
        for index in range(len(contract.inputs))
    )


def _reverse_input_scatter(contract, nodes):
    scatter = []
    for index in range(len(contract.inputs)):
        matches = [
            node_index
            for node_index, (node, _, _, _) in enumerate(nodes)
            if node.op == "load" and node.args[0] == index
        ]
        if matches:
            total = "+".join(f"g{node_index}" for node_index in matches)
            scatter.append(
                f"atomicAdjoint(inputOffsets[{index}]+pixel,upstream*({total}));"
            )
    return scatter
