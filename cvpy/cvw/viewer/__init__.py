from ._image_key_binder import ImageKeyBinder

rotation_binder = ImageKeyBinder()
rotation_binder.add_key_bind("a", rotation_binder.rotate_ccw)
rotation_binder.add_key_bind("d", rotation_binder.rotate_cw)
rotation_binder.add_key_bind("w", rotation_binder.mirror)

bbox_binder = ImageKeyBinder()
bbox_binder.show_fn = bbox_binder.bounding_box_show_fn
bbox_binder.add_key_bind("r", bbox_binder.reset_data)
