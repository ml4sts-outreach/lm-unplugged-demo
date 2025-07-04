# from svg import SVG, Polygon, Rect, Circle, ViewBoxSpec,Path
import svg
from random import random, shuffle
from numpy import linspace
from  matplotlib._color_data import XKCD_COLORS
from random import choice
import pandas as pd
from IPython.display import SVG as ipySVG


def resolve_color(color):
    if isinstance(color,str):
        if color[0] == '#':
            return color
        else:
            return XKCD_COLORS['xkcd:'+color]
    else:
        colors = [resolve_color(c) for c in color]
        
        return colors

bin_fill_color = "#eeeeee"
bin_edge_color = "#595959"
faded_edge_color ="#aaaaaa"
class Coordinate:
    def __init__(self, x,y):
        self.x = x
        self.y = y
    
    def __add__(self,change):
        if isinstance(change,tuple):
            # add to the location if a list of numbers
            return Coordinate(self.x + change[0],self.y + change[1])
        elif isinstance(change,Coordinate):
            return Coordinate(self.x + change.x,self.y + change.y)
        else:
            return NotImplemented
        
    def __sub__(self,change):
        if isinstance(change,tuple):
            # add to the location if a list of numbers
            return Coordinate(self.x - change[0],self.y - change[1])
        elif isinstance(change,Coordinate):
            return Coordinate(self.x - change.x,self.y - change.y)
        else:
            return NotImplemented
        
    def get_xy(self):
        return self.x,self.y
    
    def is_far(self,compare,dist):
        # TODO use proper distance
        if isinstance(compare,tuple):
            dx = abs(self.x-compare[0])
            dy = abs(self.y - compare[1])
            
            return (dx > dist) and (dy > dist)
        elif isinstance(compare,Coordinate):
            dx = abs(self.x-compare.x)
            dy = abs(self.y - compare.y)
            
            return (dx > dist)and (dy > dist)
        else:
            return NotImplemented
    
    def __str__(self):
        return f"pt({self.x},{self.y})"

      

class PointList:
    def __init__(self,point_list):
        self.points = point_list

    def __add__(self,to_add):
        '''
        '''
        if isinstance(to_add,tuple):
            # add to the location if a list of numbers
            dx = to_add[0]
            dy = to_add[1]
            new_points = [(xi+dx,yi+dy) for xi,yi in self.points]
            return PointList(new_points)
        elif isinstance(to_add,Coordinate):
            dx = to_add.x
            dy = to_add.y
            new_points = [(xi+dx,yi+dy) for xi,yi in self.points]
            return PointList(new_points)
        else:
            return NotImplemented
        
    def get_min_dims(self):
        w = max([xi for xi,_ in self.points])
        h = max([yi for _,yi in self.points])
        return w,h


class ImgObj:
    pad = 10

    def set_location(self,new_location):
        self.location = new_location
    
    def render_obj(self):
        # place elements before computing size
        placed_elements = self.get_elements()
        min_width, min_height = self.get_min_dims()
        
        return svg.SVG(
            viewBox=svg.ViewBoxSpec(0, 0, min_width+self.pad, min_height+self.pad),
            elements=placed_elements,)
    
    def render(self):
        return self.render_obj().as_str()
    
    def _repr_html_(self):
        return self.render_obj().as_str()
    
    def get_elements(self,container_loc = Coordinate(0,0)):
        '''
        all objects computer placement location when rendered, andonly store their 
        relative location within their container
        '''
        self.placement_loc = self.location + container_loc
        pass


class GenerateDemo(ImgObj):
    # TODO: implement a demo object, make parallel to the train demo, but for generating documents, use ball highlighting
    def __init__(self,table,doc_collection):
        table.set_location(Coordinate(0,0))
        self.table = table
        self.doc_collection = doc_collection



class TrainDemo(ImgObj):
    def __init__(self, table,doc,left=0,top=0):
        self.doc = doc
        _,doc_height = doc.get_min_dims()

        table.set_location(Coordinate(0,doc_height+100))
        self.table = table
        self.active_paths = []
        self.location = Coordinate(left,top)
        self.cur_step = 0

    def get_elements(self):
        constant_elems = self.doc.get_elements(self.location) + self.table.get_elements(self.location)
        # compute the paths after moving everything else

        path_elems = [ap.get_elements() for ap in self.active_paths]
        return constant_elems + path_elems
    
    def get_min_dims(self):
        item_far_pts = PointList([self.table.get_min_dims(),self.doc.get_min_dims()])
        return item_far_pts.get_min_dims()

    def train_step(self,word_index):
        # unlightlight others
        for word in self.doc.words:
            word.set_highlight('normal')

        for cur_bin in self.table.bins.values():
            cur_bin.set_highlight('normal')
        
        # identify active words
        prev = self.doc.get_word(word_index-1)
        prev.set_highlight('previous')
        cur = self.doc.get_word(word_index)
        cur.set_highlight('focus')
        
        # figure out bin
        self.table.add_ball(prev.name,Ball(cur.name))
        target_bin = self.table.bins[prev.name]
        target_bin.set_highlight('focus')

        # add paths
        self.active_paths = [Connector(prev,target_bin,heavy=False),Connector(cur,target_bin.contents[-1])]
        return self

    def train_next(self):
        self.cur_step += 1
        if self.cur_step >=len(self.doc.words):
            self.cur_step = 1

        return self.train_step(self.cur_step)
    
    def reset_training(self):
        self.cur_step = 0
        

class Connector(ImgObj):
    def __init__(self,source,target,heavy=True):
        self.source =source
        self.target = target
        if heavy:
            self.color = bin_edge_color
        else:
            self.color = faded_edge_color

    def get_elements(self,container_loc = Coordinate(0,0)):
        '''
        all objects 
        '''
        # self.placement_loc = self.location + container_loc
        src = self.source.get_anchor()
        tgt = self.target.get_anchor()
        

        path = svg.Path(d=[svg.M(src.x,src.y),svg.L(tgt.x,tgt.y)],stroke_width=4,stroke=self.color)
        return [path]
        

class Table(ImgObj):
    def __init__(self, bin_list,bin_spacing=5,bin_tops=0,bin_left=0):
        '''
        '''
        self.bins ={cur_bin.name:cur_bin for cur_bin in bin_list} 
        # move bins so that they do not overlap
        self.num_bins = len(bin_list)
        self.bin_spacing = bin_spacing
        self.bin_tops = bin_tops
        self.location = Coordinate(bin_left,bin_tops)
        self.placement_loc = self.location
        bin_locations = [Coordinate(x*(Bin.bin_w_top+self.bin_spacing),0) 
                                                        for x in range(self.num_bins)]
        
        for cur_bin, cur_loc in zip(self.bins.values(),bin_locations):
            cur_bin.set_location(cur_loc)

    @classmethod
    def from_list(cls, color_list):
        bin_list = [Bin(c) for c in color_list]
        return cls(bin_list)
                 
    @classmethod
    def from_csv(cls, csv_file_name):
        '''
        '''
        df = pd.read_csv(csv_file_name,index_col=0)

        bin_list = []
        for bin_color in df.index:
            ball_colors = [ci for col,ct in df.loc['purple'].to_dict().items() for ci in [col]*ct]
            shuffle(ball_colors)
            cur_contents = [Ball(ci) for ci in ball_colors]
            bin_list.append(Bin(bin_color,contents=cur_contents))

        return cls(bin_list)

    def set_location(self,new_location,):
        self.location =new_location

    def get_elements(self,container_loc = Coordinate(0,0)):
        
        self.placement_loc = self.location +container_loc
        # print('rendering bins relative to table ',self.placement_loc)
        return  [ei for bin in self.bins.values() for ei in bin.get_elements(self.placement_loc)]
    
    def get_min_dims(self):
        container_adjust = self.placement_loc - self.location
        item_far_pts = PointList([it.get_min_dims() for it in self.bins.values()]) + container_adjust
        return item_far_pts.get_min_dims()
    
    def add_ball(self,bin_name,ball):
        self.bins[bin_name].add_ball(ball)
        return self.bins[bin_name].contents[-1]
    
    def sample_bin(self,name):
        
        return self.bins[name].sample()

    def sample_doc(self,prompt):
        sampled_doc = Doc.from_list([prompt],max_width_words=5)
        last_word = prompt
        while not(last_word =='white'):
            sampled_word = self.sample_bin(last_word)
            
            sampled_doc.add_word(sampled_word)
            last_word = sampled_word

        return sampled_doc
    
    

class Bin(ImgObj):
    # TODO: figure out how to make it possible to train with more balls
    bin_w_top = 140
    bin_bottom_offset = round(.25*bin_w_top)
    bin_h = 160
    sticky_width = 30
    sticky_height = 20
    sticky_offset = 10
    pad = 5
    stroke_color_highlight = {'normal':bin_edge_color,
                        'previous':bin_edge_color,
                        'focus':bin_edge_color}
    stroke_width_highlight = {'normal':1,
                              'previoius':2,
                        'focus':4}
    def __init__(self,color,left_x=0,top_y=0,contents=None,highlight='normal'):
        # Define dimensions and central positions for bin components
        self.color = resolve_color(color)
        self.highlight = highlight
        if isinstance(color,list):
            self.n_labels = len(color)
            self.name = '-'.join(color)
        else:
            self.n_labels = 1
            self.name = color

        self.label = Label(self.color)
        self.location = Coordinate(left_x,top_y)
        self.placement_loc = self.location
        self.base_points = PointList([(0,0), (Bin.bin_bottom_offset, Bin.bin_h), 
                                 (Bin.bin_w_top-Bin.bin_bottom_offset, Bin.bin_h), (Bin.bin_w_top, 0)]) 
        

        self.contents = []
        self.compute_coodinates()
        if contents:
            # place balls relatively
            for ball in contents:
                ball.set_location(self.coordinates[len(self.contents)])
                self.contents.append(ball)
                # self.add_ball(ball)
        
    def get_anchor(self):
        return self.placement_loc + Coordinate(self.bin_w_top/2,0)

    def set_focus(self):
        self.highlight = 'focus'
    
    def set_previous(self):
        self.highlight = 'previous'

    def set_normal(self):
        self.highlight = 'normal'

    def set_highlight(self,new_highlight):
        self.highlight = new_highlight

    def get_elements(self,container_loc=Coordinate(0,0),highlight=None):
        # if passed, use but do not set
        if not(highlight):
            highlight = self.highlight

        # relative move only on render
        self.placement_loc = self.location + container_loc      
        
        bin_points = self.base_points + self.placement_loc
        # Define the main bin polygon and top rectangle part
        bin_polygon = svg.Polygon(points=bin_points.points, 
                    fill=bin_fill_color,
                    stroke=self.stroke_color_highlight[highlight],
                    stroke_width=self.stroke_width_highlight[highlight])
        
        # Todo: still needs to be expanded for multiple
        sticky_loc = self.placement_loc + (Bin.sticky_offset,0)
        sticky = self.label.get_elements(sticky_loc)
        
        # print('placing balls relative to ',self.placement_loc)
        # move balls relatively and get their elements
        contents = [item.get_elements(self.placement_loc) for item in self.contents]
        
        return [bin_polygon,sticky] +contents
    
    def get_min_dims(self):
        bin_points = self.base_points + self.placement_loc
        return bin_points.get_min_dims()
    
    def sample(self):
        return choice(self.contents).name
        

    def add_ball(self,ball):
        # set ball's relative location only
        ball.set_location(self.coordinates[len(self.contents)])
        self.contents.append(ball)

    
    def compute_coodinates(self):
        width_diff = 2*Bin.bin_bottom_offset
        center_min_width = Bin.bin_w_top -width_diff - 2*Bin.pad -2*Ball.radius
        usable_min_width = Bin.bin_w_top -2*Bin.bin_bottom_offset - 2*Bin.pad
        left_at_height = lambda y: y*Bin.bin_bottom_offset + Ball.radius + Bin.pad
        usable_width_at_height = lambda y: (1-y)*width_diff + usable_min_width
        center_width_at_height = lambda y: (1-y)*width_diff + center_min_width
        usable_height = Bin.bin_h-(Bin.pad+Bin.sticky_height)
        center_usable_height = usable_height-2*Ball.radius
        max_vert_balls = usable_height//(2*(Ball.radius+Ball.pad))
        # bias to bottom
        balls_at_height = lambda y: int(usable_width_at_height(y)//(2*(Ball.radius+Ball.pad)))

        coords = [[Coordinate(left_at_height(rel_y)+x,rel_y*center_usable_height+Bin.pad+Bin.sticky_height)
                                for x in linspace(0,center_width_at_height(rel_y),num=balls_at_height(rel_y))]
                                            for rel_y in linspace(1,0,max_vert_balls) ]
        [shuffle(row) for row in coords]
        # TODO:  should be jittered too
        self.coordinates = [c for row in coords for c in row]
        
    def __str__(self):
        return f"bin at {self.location}"
        
    @staticmethod
    def ball_loc_candidate(prev_placed):

        width_diff = 2*Bin.bin_bottom_offset
        min_width = Bin.bin_w_top -2*Bin.bin_bottom_offset - 2*Bin.pad -Ball.radius
        left_at_height = lambda y: y*Bin.bin_bottom_offset + Ball.radius + Bin.pad
        width_at_height = lambda y: (1-y)*width_diff + min_width
        usable_height = Bin.bin_h-(2*Ball.radius+Bin.pad+Bin.sticky_height)
        # random location within the height
        rel_y = random()
        candidate_y = rel_y*usable_height +Bin.pad + Ball.radius+Bin.sticky_height

        # random location within the width at that height 
        rel_x = random()
        candidate_x = rel_x*width_at_height(candidate_y) + left_at_height(candidate_y)
        return Coordinate(round(candidate_x),round(candidate_y))
    
    @staticmethod
    def get_ball_loc_ordered(self,prev_placed):
        width_diff = 2*Bin.bin_bottom_offset
        center_min_width = Bin.bin_w_top -2*Bin.bin_bottom_offset - 2*Bin.pad -Ball.radius
        usable_min_width = Bin.bin_w_top -2*Bin.bin_bottom_offset - 2*Bin.pad
        left_at_height = lambda y: y*Bin.bin_bottom_offset + Ball.radius + Bin.pad
        width_at_height = lambda y: (1-y)*width_diff + usable_min_width
        usable_height = Bin.bin_h-(Bin.pad+Bin.sticky_height)
        center_usable_height = usable_height-2*Ball.radius
        max_vert_balls = usable_height//(2*(Ball.radius+Ball.pad))
        # bias to bottom
        balls_at_height = lambda y: width_at_height(y)//(2*(Ball.radius+Ball.pad))
        # prev_bias = (1-prev_placed/)


class DocCollection(ImgObj):
    def __init__(self,doc_list=None):
        # TODO: create a container that can hold mutliple documents,fill in the elements and other methods needed
        self.doc_list = []
        if doc_list:
            for doc in doc_list:

                doc.set_location(self.get_next_doc_loc())
                self.doc_list.append(doc)
    
    def get_next_doc_loc(self):
        # calculate, probably align vertically
        return Coordinate(0,0)

class Doc(ImgObj):
    sticky_width = 100
    sticky_height = 60
    def __init__(self, sticky_list,word_spacing=10,left =0,top=0,
            max_width_words = None,end_token='#ffffff'):
        '''
        '''
        # self.words = sticky_list 
        # move stickies so that they do not overlap
        self.end_token = end_token
        self.word_spacing = word_spacing
        self.location = Coordinate(left,top)
        self.placement_loc = self.location
        if max_width_words:
            self.word_wrap = True
            self.words_per_row = max_width_words
        else:
            self.word_wrap = False
        
        self.words = []
        for cur_word in sticky_list:
            self.add_word(cur_word)
    
    @classmethod
    def from_list(cls,word_list,max_width_words=None):
        sticky_list = [Sticky(word) for word in word_list]
        return cls(sticky_list,max_width_words=max_width_words)
    
    @classmethod
    def from_string(cls,doc_string,max_width_words=None):
        sticky_list = [Sticky(word) for word in doc_string.split()]
        return cls(sticky_list,max_width_words=max_width_words)
    
    def is_valid(self):
        if self.words:
            return self.words[-1].color == self.end_token
        else:
            # if empty, always valid
            return True
    
    def reset_words(self,new_words):
        self.words = []
        for cur_word in new_words:
            self.add_word(cur_word)

    def get_word(self,index):
        return self.words[index]
    
    def get_elements(self,container_loc = Coordinate(0,0)):
        self.placement_loc = self.location + container_loc
        return  [ei for it in self.words for ei in it.get_elements(self.placement_loc)]
    
    def get_min_dims(self):
        # compute size of set of stickies
        num_words = len(self.words)
        if self.word_wrap:
            # use fixed width, adjust height
            width = self.words_per_row*(self.sticky_width+self.word_spacing)
            rows = num_words//self.words_per_row +1
            height = rows*(self.sticky_height+self.word_spacing) +self.word_spacing
            
        else:
            # adjust width, constant height
            width = num_words*(self.sticky_width+self.word_spacing)
            height = self.sticky_height+2*self.word_spacing
        # add doc location as offset
        far_points = PointList([(width,height)])+ self.placement_loc
        return far_points.get_min_dims()
    
    def add_word(self,to_add):
        # can add by color creating a new or add an already created sticky
        if isinstance(to_add,str):
            new_word = Sticky(to_add)
        elif isinstance(to_add,Sticky):
            new_word = to_add
        
        new_word.set_location(self.get_next_word_loc())
        self.words.append(new_word)
        
        return self
    

    def get_next_word_loc(self):
        cur_word_num = len(self.words) 
        if self.word_wrap:
            row = cur_word_num//self.words_per_row
            position_in_row = cur_word_num%self.words_per_row
            return Coordinate(position_in_row*(self.sticky_width+self.word_spacing),
                              row*(self.sticky_height+self.word_spacing))
        else:
            # jsut wide foreverrrrrr
            return Coordinate((cur_word_num)*(self.sticky_width+self.word_spacing),0)
    



class Word:
    # TODO: add more shapes here
    symbol_shape_svg = {'star':PointList([(30,6), (36.6,22.8), (54,22.8), (39.6,35.4), (45,53) ,(30,42) ,
                                          (15,53), (20.4,35.4),( 6,22.8), (23.4,22.8)])}
    def __init__(self,color,symbol=''):
        self.color = resolve_color(color)
        self.symbol = symbol
        self.name = color + symbol

    def get_color(self):
        return self.color
    
    def get_sticky_shape(self,target):
        if self.symbol:
            # do stuff
            return self.symbol_shape_svg[self.symbol]
        else:
            return target.default_shape 

    def get_ball_decor(self,target):  
        return 

class Sticky(ImgObj):
    stroke_color_highlight = {'normal':bin_fill_color,
                        'previous':faded_edge_color,
                        'focus':bin_edge_color}
    stroke_width_highlight = {'normal':1, 
                              'previous':4,
                        'focus':4}
    default_shape = svg.Rect
    def __init__(self,color,left_x=0,top_y=0,width=Doc.sticky_width,height=Doc.sticky_height,
                 highlight='normal'):
        # Define dimensions and central positions for bin components
        self.color = resolve_color(color)
        self.name = color
        self.width = width
        self.height = height
        self.location = Coordinate(left_x,top_y)
        self.placement_loc = self.location
        self.highlight = highlight
        
    def get_anchor(self):        
        return self.placement_loc + Coordinate(self.width/2,self.height)
    
    def set_focus(self):
        self.highlight = 'focus'
    
    def set_previous(self):
        self.highlight = 'previous'

    def set_normal(self):
        self.highlight = 'normal'

    def set_highlight(self,new_highlight):
        self.highlight = new_highlight
    
    def get_elements(self,container_loc=Coordinate(0,0),highlight=None):
        # highlight at render does not set attribute
        if not(highlight):
            highlight = self.highlight
        # relative move only on render
        self.placement_loc = self.location + container_loc       
        x,y = self.placement_loc.get_xy() 

        # TODO: enable sticky class to use the 'word' class, getting the shape here from there
        sticky = svg.Rect(x=x,y=y,
                      width=self.width, height=self.height, fill=self.color, 
                      stroke=self.stroke_color_highlight[highlight],
                      stroke_width=self.stroke_width_highlight[highlight])
        
        return [sticky] 

class Label(Sticky):
    # TODO: implement label and use it for bins
    def __init__(self, color, left_x=0, top_y=0, width=Bin.sticky_width, height=Bin.sticky_height):
        super().__init__(color, left_x, top_y, width, height)


class LabelGroup(ImgObj):
    # TODO: implement labelgroup to hold two labls and allow for bins to be created for longer context windows
    def __init__(self):
        super().__init__()

class Ball(ImgObj):
    # TODO: implement ball highlights (follow  sticky and bin as examples)
    radius = 10
    pad = 2
    def __init__(self,color,cx=radius,cy=radius):
        '''
        '''
        self.location = Coordinate(cx,cy)
        self.placement_loc = self.location
        self.color = resolve_color(color)
        self.name = color
        
    
    def adjust(self,rel_location):
        self.location +=rel_location

    def get_anchor(self):
        
        return self.placement_loc + Coordinate(0,-self.radius)
       
    
    def get_min_dims(self):
        far_loc = self.location + Coordinate(Ball.radius,Ball.radius)
        return far_loc.get_xy()
    
    def get_elements(self,container_loc=Coordinate(0,0)):
        # relative move only on render
        self.placement_loc = self.location + container_loc       
        x,y = self.placement_loc.get_xy()
        
        # TODO: make balls able to use word class to give a decoration here
        ball = svg.Circle(
                cx=x, cy=y, r=Ball.radius,
                fill=self.color,
                stroke="transparent",
            )
        
        return [ball]
    
    def is_far(self,candidate):
        return self.location.is_far(candidate,self.radius+1)
    
    def __str__(self):
        return f"{self.color} ball at {self.location}"




  