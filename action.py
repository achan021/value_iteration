#dictionary of actions 
#'U','D','L','R'

actions = {
(0,0) : ('D'),
(2,0) : ('D','R'),
(3,0) : ('D','L','R'),
(4,0) : ('L','R'),
(5,0) : ('D','L'),
(0,1) : ('U','D','R'),
(1,1) : ('D','L','R'),
(2,1) : ('U','D','L','R'),
(3,1) : ('U','D','L'),
(5,1) : ('U','D'),
(0,2) : ('U','D','R'),
(1,2) : ('U','D','L','R'),
(2,2) : ('U','D','L','R'),
(3,2) : ('U','D','L','R'),
(4,2) : ('D','L','R'),
(5,2) : ('U','D','L'),
(0,3) : ('U','D','R'),
(1,3) : ('U','L','R'),
(2,3) : ('U','L','R'),
(3,3) : ('U','L','R'),
(4,3) : ('U','D','L','R'),
(5,3) : ('U','D','L'),
(0,4) : ('U','D'),
(4,4) : ('U','D','R'),
(5,4) : ('U','D','L'),
(0,5) : ('U','R'),
(1,5) : ('L','R'),
(2,5) : ('L','R'),
(3,5) : ('L','R'),
(4,5) : ('U','L','R'),
(5,5) : ('U','L')
}


