import os
import shutil

import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

if not os.path.exists("user_custom_layout.ini"):
    shutil.copy("custom_layout.ini", "user_custom_layout.ini")

dpg.create_context()
dpg.configure_app(docking=True, docking_space=True, init_file="user_custom_layout.ini")
dpg.create_viewport(title='SDF UI Editor', width=800, height=600)
dpg.setup_dearpygui()

center_window = dpg.generate_uuid()
settings_window = dpg.generate_uuid()

with dpg.font_registry():
    default_font = dpg.add_font("OpenSans-Regular.ttf", 18)
    dpg.bind_font(default_font)


def link_callback(sender, app_data):
    dpg.add_node_link(app_data[0], app_data[1], parent=sender)
    print(f"Connection established {app_data[0], app_data[1]}")


def delink_callback(sender, app_data):
    dpg.delete_item(app_data)
    print("Connection terminated")


def print_me(sender):
    print(f"Menu Item: {sender}")


with dpg.viewport_menu_bar():
    with dpg.menu(label="File"):
        dpg.add_menu_item(label="Save", callback=print_me)
        dpg.add_menu_item(label="Save As", callback=print_me)

        with dpg.menu(label="Settings"):
            dpg.add_menu_item(label="Setting 1", callback=print_me, check=True)
            dpg.add_menu_item(label="Setting 2", callback=print_me)

    dpg.add_menu_item(label="Help", callback=print_me)

    with dpg.menu(label="Widget Items"):
        dpg.add_checkbox(label="Pick Me", callback=print_me)
        dpg.add_button(label="Press Me", callback=print_me)
        dpg.add_color_picker(label="Color Me", callback=print_me)

with dpg.window(label="Node Editor", tag=center_window, no_collapse=True, no_close=True, no_title_bar=True,
                no_background=True, no_move=True):
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Save", callback=print_me)
            dpg.add_menu_item(label="Save As", callback=print_me)
            dpg.add_checkbox(label="Tutorial", callback=lambda x: print("Hello World"))

            with dpg.menu(label="Settings"):
                dpg.add_menu_item(label="Setting 1", callback=print_me, check=True)
                dpg.add_menu_item(label="Setting 2", callback=print_me)

        dpg.add_menu_item(label="Help", callback=print_me)

        with dpg.menu(label="Widget Items"):
            dpg.add_checkbox(label="Pick Me", callback=print_me)
            dpg.add_button(label="Press Me", callback=print_me)
            dpg.add_color_picker(label="Color Me", callback=print_me)

    with dpg.node_editor(callback=link_callback, delink_callback=delink_callback):
        with dpg.node(label="Node 1"):
            with dpg.node_attribute(label="Node A1"):
                dpg.add_input_float(label="F1", width=150)

            with dpg.node_attribute(label="Node A2", attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(label="F2", width=150)

        with dpg.node(label="Node 2"):
            with dpg.node_attribute(label="Node A3"):
                dpg.add_input_float(label="F3", width=200)

            with dpg.node_attribute(label="Node A4", attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(label="F4", width=200)


demo.show_demo()

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
