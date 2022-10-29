% allow_singletons.

head_pred(kp,1).
body_pred(consists_of_shape,3).
body_pred(consists_of_color,3).
body_pred(consists_of_large_objects,2).
body_pred(number_large_objects,2).
body_pred(is_no_member,2).
body_pred(not_equal,2).


type(kp,(kf,)).
type(consists_of_shape,(kf,comp,list)).
type(consists_of_color,(kf,comp,list)).
type(consists_of_large_objects,(kf,list)).
type(number_large_objects,(kf,num)).
type(is_no_member,(comp,list)).
type(not_equal,(comp,comp)).

direction(kp,(in,)).
direction(consists_of_shape,(in,out,out)).
direction(consists_of_color,(in,out,out)).
direction(consists_of_large_objects,(in,out)).
direction(number_large_objects,(in,out)).
direction(is_no_member,(in,in)).
direction(not_equal,(in,in)).