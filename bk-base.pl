% lists of elements belonging to large figures or large figures to kf
consists_of_shape(Kf, Comp, List) :- setof(Small, part_of_shape(Kf, Small, Comp), List). 
consists_of_color(Kf, Comp, List) :- setof(Color, part_of_color(Kf, Color, Comp), List). 
consists_of_large_objects(Kf, List) :- setof(Comp, part_of_comp(Kf, Comp), List).

part_of_shape(Kf, Small, Comp) :- part_of(Kf,_,Small,Comp,_,_,_,_).
part_of_color(Kf, Color, Comp) :- part_of(Kf,Color,_,Comp,_,_,_,_).
part_of_comp(Kf, Comp) :- part_of(Kf,_,_,Comp,_,_,_,_).

% counting
number_large_objects(Kf, Num) :- consists_of_large_objects(Kf, List), length(List, Num).

% general knowledge about composite objects and equality:
is_member(A, List) :- member(A, List).
is_no_member(A, List) :- not(member(A, List)).
not_equal(A, B) :- A \= B.
