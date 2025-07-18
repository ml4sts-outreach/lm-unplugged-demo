 # from svg import SVG, Polygon, Rect, Circle, ViewBoxSpec,Path
import svg
from random import random, shuffle
from numpy import linspace
from  matplotlib._color_data import XKCD_COLORS
from random import choice
import pandas as pd
from IPython.display import SVG as ipySVG
from collections import Counter


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




class TrainDemo(ImgObj):
    def __init__(self, table,doc,left=0,top=0):
        self.doc = doc
        _,doc_height = doc.get_min_dims()
        table.set_location(Coordinate(0,doc_height+100))
        self.table = table

        # set doc context from table context
        self.doc.set_context(table.context_len)

        
        self.active_paths = []
        self.location = Coordinate(left,top)
        self.cur_step = self.doc.context_len-1

    def get_elements(self):
        constant_elems = self.doc.get_elements(self.location) + self.table.get_elements(self.location)
        # compute the paths after moving everything else

        path_elems = [ap.get_elements() for ap in self.active_paths]
        return constant_elems + path_elems
    
    def get_min_dims(self):
        item_far_pts = PointList([self.table.get_min_dims(),self.doc.get_min_dims()])
        return item_far_pts.get_min_dims()

    def train_step(self,word_index):
        # unlightlight all before starting
        for word in self.doc.words:
            word.set_highlight('normal')

        for cur_bin in self.table.bins.values():
            cur_bin.set_highlight('normal')
            for ball in cur_bin.contents:
                ball.set_highlight('normal')
        
        # identify active words
              
        prev = self.doc.get_context(word_index)
        [previ.set_highlight('previous') for previ in prev]
        cur = self.doc.get_word(word_index)
        cur.set_highlight('focus')
        
        # add ball to bin and highlight them
        
        target_bin_name = '-'.join([p.name for p in prev])
        self.table.add_ball(target_bin_name,Ball(cur.name))
        target_bin = self.table.bins[target_bin_name]
        target_bin.set_highlight('focus')
        target_bin.contents[-1].set_highlight('focus')

        # add paths, just connect last word incontext window, both will be higlighted
        prev_paths = [Connector(word,label,heavy=False) for word,label in zip(prev,target_bin.label.words)]
        self.active_paths = prev_paths + [Connector(cur,target_bin.contents[-1])]
        return self

    def train_next(self):
        self.cur_step += 1
        if self.cur_step >=len(self.doc.words):
            return self.reset_training()
        else:
            return self.train_step(self.cur_step)
    
    def reset_training(self):
        self.active_paths = []
        for word in self.doc.words:
            word.set_highlight('normal')

        for cur_bin in self.table.bins.values():
            cur_bin.set_highlight('normal')
            for ball in cur_bin.contents:
                ball.set_highlight('normal')

        self.cur_step = self.doc.context_len-1
        return self
        


class SampleDemo(ImgObj):
    def __init__(self, table,doc,left=0,top=0):

        self.doc = doc
        _,doc_height = doc.get_min_dims()

        table.set_location(Coordinate(0,doc_height+100))
        self.table = table
        _,table_height = table.get_min_dims()

        # TODO: add collection and make it possible to sample multiple documents in a row
        self.active_paths = []
        self.location = Coordinate(left,top)
        self.cur_step = 0
        self.doc.set_context(table.context_len)

    def get_elements(self):
        constant_elems = self.doc.get_elements(self.location) + self.table.get_elements(self.location)
        # compute the paths after moving everything else

        path_elems = [ap.get_elements() for ap in self.active_paths]
        return constant_elems + path_elems
    
    def get_min_dims(self):
        item_far_pts = PointList([self.table.get_min_dims(),self.doc.get_min_dims()])
        return item_far_pts.get_min_dims()

    def sample_step(self):
        # always reset by unhiglighting and removing paths
        for word in self.doc.words:
            word.set_highlight('normal')

        for cur_bin in self.table.bins.values():
            cur_bin.set_highlight('normal')
            for ball in cur_bin.contents:
                ball.set_highlight('normal')

        
        self.active_paths = []
        
        
        last_word = self.doc.words[-1]
        # keep sampling as long as the last word is not white
        if not(last_word.name == 'white'):

            context = self.doc.get_context(-1)
            target_bin_name = '-'.join([p.name for p in context])
            target_bin = self.table.bins[target_bin_name]
            
            target_bin.set_highlight('previous')

            # sample new word
            sampled_ball = target_bin.sample(return_object=True)
            sampled_ball.set_highlight('focus')
            self.doc.add_word(sampled_ball.name)
            new_word = self.doc.words[-1]

        
            # identify active words
            
            [previ.set_highlight('previous') for previ in context]
            # self.doc.add_word(new_word)
            new_word.set_highlight('focus')
        
        

            # add paths
            prev_paths = [Connector(word,label,heavy=False) for word,label in 
                          zip(context,target_bin.label.words)]
            self.active_paths = prev_paths + [Connector(sampled_ball,new_word)]
        return self

    
    def reset_sampling(self):
        self.active_paths = []
        for word in self.doc.words:
            word.set_highlight('normal')

        for cur_bin in self.table.bins.values():
            cur_bin.set_highlight('normal')
            for ball in cur_bin.contents:
                ball.set_highlight('normal')

        return self


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
        src = self.source.get_anchor(type='bottom')
        tgt = self.target.get_anchor(type='top')
        

        path = svg.Path(d=[svg.M(src.x,src.y),svg.L(tgt.x,tgt.y)],stroke_width=4,stroke=self.color)
        return [path]
        

class Table(ImgObj):
    def __init__(self, bin_list,bin_spacing=10,bin_tops=0,bin_left=0,max_width_bins=None):
        '''
        '''
        self.bins ={cur_bin.name:cur_bin for cur_bin in bin_list}
        # TODO: valdate that all have the same number of labels 
        bin_names = list(self.bins.keys())
        self.context_len = len(bin_names[0].split('-'))
        # move bins so that they do not overlap
        self.num_bins = len(bin_list)
        self.bin_spacing = bin_spacing
        self.bin_tops = bin_tops
        self.location = Coordinate(bin_left,bin_tops)
        self.placement_loc = self.location
        
        if max_width_bins:
            self.bin_wrap = True
            self.bins_per_row = max_width_bins
        else:
            self.bin_wrap = False

        # bin_locations = [Coordinate(x*(bin_w_top+self.bin_spacing),0) 
        #                                                 for x in range(self.num_bins)]
        
        for i,cur_bin in enumerate(self.bins.values()):
            cur_bin.set_location(self.get_bin_loc_by_index(i))
            

    @classmethod
    def from_list(cls, color_list,max_width_bins=None):
        bin_list = [Bin(c) for c in color_list]
        return cls(bin_list,max_width_bins=max_width_bins)
                 
    @classmethod
    def from_csv(cls, csv_file_name,max_width_bins=None,bin_size='small'):
        '''
        create table from csv file with previous as index and possible next as columns
        index columns should have no column names for context >1 we need a multiindex
        counts may be scaled if too many balls to visualize
        '''
        df = pd.read_csv(csv_file_name)
        un = [c for c in df.columns if 'unnamed' in c.lower()]
        df = df.set_index(un)
        max_balls_per_bin = len(Bin('white',bin_size=bin_size).coordinates)

        bin_list = []
        for bin_color in df.index:
            ball_counts = df.loc[bin_color].to_dict().items()
            bc_sum = df.loc[bin_color].sum()
            if bc_sum >= max_balls_per_bin:
                
                # if more balls than can fit, scale down
                scale_factor = .8*max_balls_per_bin/df.loc[bin_color].sum()
                ball_counts = [(col,round(ct*scale_factor)) for col,ct in ball_counts]
                
                
            ball_colors = [ci for col,ct in ball_counts for ci in [col]*ct]
            shuffle(ball_colors)
            cur_contents = [Ball(ci) for ci in ball_colors]
            
            # make labelgroup or label
            if isinstance(df.index,pd.MultiIndex):
                bin_color = list(bin_color)
                
            bin_list.append(Bin(bin_color,contents=cur_contents,bin_size=bin_size))

        return cls(bin_list,max_width_bins=max_width_bins)

    def set_location(self,new_location,):
        self.location =new_location

    def get_elements(self,container_loc = Coordinate(0,0)):
        
        self.placement_loc = self.location +container_loc
        # print('rendering bins relative to table ',self.placement_loc)
        return  [ei for bin in self.bins.values() for ei in bin.get_elements(self.placement_loc)]
    
    def get_df(self):
        bin_names = list(self.bins.keys())
        bin_stickies = [b for b in bin_names]
        
        bin_series = [bin.get_series() for _, bin in self.bins.items()]
        
        return pd.DataFrame(data=bin_series, index= bin_stickies)
    
    def get_min_dims(self):
        container_adjust = self.placement_loc - self.location
        item_far_pts = PointList([it.get_min_dims() for it in self.bins.values()]) + container_adjust
        return item_far_pts.get_min_dims()
    
    def add_ball(self,bin_name,ball):
        self.bins[bin_name].add_ball(ball)
        return self.bins[bin_name].contents[-1]
    
    def sample_bin(self,name,return_object=False):
        
        return self.bins[name].sample(return_object=return_object)

    def sample_doc(self,prompt,max_width_words=5):
        if not isinstance(prompt,list):
            prompt = [prompt]

        sampled_doc = Doc.from_list(prompt,max_width_words=max_width_words,
                                    context_len=self.context_len)
        
        last_word = prompt[-1]
        while not(last_word =='white'):
            context = sampled_doc.get_context(-1)
            bin_name = '-'.join([c.name for c in context])
            sampled_word = self.sample_bin(bin_name)
            
            sampled_doc.add_word(sampled_word)
            last_word = sampled_word

        return sampled_doc
    
    def get_bin_loc_by_index(self,cur_bin_num):
        bin_w_top = max([b.bin_w_top for b in self.bins.values()])
        bin_h = max([b.bin_h for b in self.bins.values()])
        if self.bin_wrap:
            row = cur_bin_num//self.bins_per_row
            position_in_row = cur_bin_num%self.bins_per_row
            return Coordinate(position_in_row*(bin_w_top+self.bin_spacing),
                              row*(bin_h+self.bin_spacing))
        else:
            # jsut wide foreverrrrrr
            return Coordinate((cur_bin_num)*(bin_w_top+self.bin_spacing),0)
    
small_bin_size = {'bin_w_top':140,
    'bin_bottom_offset' :35,
    'bin_h' : 160}    

large_bin_size = {'bin_w_top':180,
    'bin_bottom_offset' :27,
    'bin_h' : 205}  

bin_sizes = {'small':small_bin_size,
             'large':large_bin_size}

class Bin(ImgObj):
    # TODO: expand more sizes or ways to let balls overlap
    sticky_width = 30
    sticky_height = 20
    sticky_offset = 10
    pad = 5
    stroke_color_highlight = {'normal':bin_edge_color,
                        'previous':bin_edge_color,
                        'focus':bin_edge_color}
    stroke_width_highlight = {'normal':1,
                              'previous':2,
                        'focus':4}
    def __init__(self,color,left_x=0,top_y=0,contents=None,highlight='normal',
                 bin_size = 'small'):
        # set bin size
        for dim, val in bin_sizes[bin_size].items():
            setattr(self,dim,val)    

        self.highlight = highlight
        if isinstance(color,list):
            self.n_labels = len(color)
            self.name = '-'.join(color)
            self.label = LabelGroup([Label(c) for c in color])
        else:
            self.n_labels = 1
            self.name = color
            self.label = Label(color)

            # self.color = resolve_color(color)

        
        self.location = Coordinate(left_x,top_y)
        self.placement_loc = self.location
        self.base_points = PointList([(0,0), (self.bin_bottom_offset, self.bin_h), 
                                 (self.bin_w_top-self.bin_bottom_offset, self.bin_h), (self.bin_w_top, 0)]) 
        

        self.contents = []
        self.compute_coodinates()
        if contents:
            # place balls relatively
            for ball in contents:
                ball.set_location(self.coordinates[len(self.contents)])
                self.contents.append(ball)
                # self.add_ball(ball)
        
    def get_anchor(self,type=None):
        return self.placement_loc + Coordinate(self.bin_w_top/2,0)
    
    def get_series(self):
        # count the contents
        ball_colors = [ball.name for ball in self.contents]
        return pd.Series(Counter(ball_colors))

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
        
        
        sticky_loc = self.placement_loc + (Bin.sticky_offset,0)
        sticky = self.label.get_elements(sticky_loc)
        
        # print('placing balls relative to ',self.placement_loc)
        # move balls relatively and get their elements
        contents = [item.get_elements(self.placement_loc) for item in self.contents]
        
        return [bin_polygon,sticky] +contents
    
    def get_min_dims(self):
        bin_points = self.base_points + self.placement_loc
        return bin_points.get_min_dims()
    
    def sample(self,return_object=False):
        '''
        '''
        if return_object:
            return choice(self.contents)
        else:
            return choice(self.contents).name
        

    def add_ball(self,ball):
        # set ball's relative location only
        if not isinstance(ball,Ball):
            # if not a ball, create one
            ball = Ball(ball)
        ball.set_location(self.coordinates[len(self.contents)])
        self.contents.append(ball)
        

    
    def compute_coodinates(self):
        width_diff = 2*self.bin_bottom_offset
        center_min_width = self.bin_w_top -width_diff - 2*self.pad -2*Ball.radius
        usable_min_width = self.bin_w_top -2*self.bin_bottom_offset - 2*self.pad
        left_at_height = lambda y: y*self.bin_bottom_offset + Ball.radius + self.pad
        usable_width_at_height = lambda y: (1-y)*width_diff + usable_min_width
        center_width_at_height = lambda y: (1-y)*width_diff + center_min_width
        usable_height = self.bin_h-(self.pad+self.sticky_height)
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
        
    # @staticmethod - unused
    def ball_loc_candidate(self,prev_placed):

        width_diff = 2*self.bin_bottom_offset
        min_width = self.bin_w_top -2*self.bin_bottom_offset - 2*Bin.pad -Ball.radius
        left_at_height = lambda y: y*self.bin_bottom_offset + Ball.radius + Bin.pad
        width_at_height = lambda y: (1-y)*width_diff + min_width
        usable_height = self.bin_h-(2*Ball.radius+Bin.pad+Bin.sticky_height)
        # random location within the height
        rel_y = random()
        candidate_y = rel_y*usable_height +Bin.pad + Ball.radius+Bin.sticky_height

        # random location within the width at that height 
        rel_x = random()
        candidate_x = rel_x*width_at_height(candidate_y) + left_at_height(candidate_y)
        return Coordinate(round(candidate_x),round(candidate_y))
    
    @staticmethod
    # not used
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
        # calculate, probably align vertically but with wrapping optionally
        return Coordinate(0,0)


class StickyContainer(ImgObj):
    sticky_width = 100
    sticky_height = 60
    def __init__(self, sticky_list,word_spacing=10,left =0,top=0,
            max_width_words = None,end_token='#ffffff',context_len=1):
        '''
        '''
        # self.words = sticky_list 
        # move stickies so that they do not overlap
        self.end_token = end_token
        self.word_spacing = word_spacing
        self.location = Coordinate(left,top)
        self.placement_loc = self.location
        self.context_len = context_len
        if max_width_words:
            self.word_wrap = True
            self.words_per_row = max_width_words
        else:
            self.word_wrap = False
        
        self.words = []
        for cur_word in sticky_list:
            self.add_word(cur_word)
    
    @classmethod
    def from_list(cls,word_list,max_width_words=None,context_len=1):
        sticky_list = [Sticky(word) for word in word_list]
        return cls(sticky_list,max_width_words=max_width_words,context_len=context_len)
    
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
    
    def get_context(self,index):
        # if -1 should return the last self.context_len words
        # if positive shoudl return self.context_len words before the index including the index
        if index < 0:  
            return self.words[-self.context_len:]
        else:
            return self.words[index-self.context_len:index]
    
    def set_context(self,new_context_len):
        self.context_len = new_context_len
    
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
    

class Doc(StickyContainer):
    sticky_width = 100
    sticky_height = 60

class MiniDoc(StickyContainer):
    sticky_width = 50
    sticky_height = 30


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
                              'previous':5,
                        'focus':5}
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
        
    def get_anchor(self,type=None):
        if type == 'top':
            return self.placement_loc + Coordinate(self.width/2,0)
        elif type == 'bottom':        
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
    
    def __str__(self):
        return self.get_elements()[0].as_str()

class Label(Sticky):
    def __init__(self, color, left_x=0, top_y=0, width=Bin.sticky_width, height=Bin.sticky_height):
        super().__init__(color, left_x, top_y, width, height)


class LabelGroup(StickyContainer):

    sticky_width = Bin.sticky_width
    sticky_height = Bin.sticky_height
    def __init__(self, labels):
        context_length = len(labels)
        super().__init__(labels,context_len=context_length,
                         max_width_words=context_length,)

class Ball(ImgObj):
    radius = 10
    pad = 2

    
    stroke_color_highlight = {'normal':"transparent",
                        'previous':bin_edge_color,
                        'focus':bin_edge_color}
    stroke_width_highlight = {'normal':1,
                              'previous':2,
                        'focus':4}
    def __init__(self,color,cx=radius,cy=radius,highlight='normal'):
        '''
        '''
        self.location = Coordinate(cx,cy)
        self.placement_loc = self.location
        self.color = resolve_color(color)
        self.name = color
        self.highlight = highlight
        
    
    def adjust(self,rel_location):
        self.location +=rel_location

    def get_anchor(self,type=None):
        
        return self.placement_loc + Coordinate(0,-self.radius)
       
    
    def get_min_dims(self):
        far_loc = self.location + Coordinate(Ball.radius,Ball.radius)
        return far_loc.get_xy()
    

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
        x,y = self.placement_loc.get_xy()
        
        # TODO: make balls able to use word class to give a decoration here
        ball = svg.Circle(
                cx=x, cy=y, r=Ball.radius,
                fill=self.color,
                stroke=self.stroke_color_highlight[highlight],
                stroke_width=self.stroke_width_highlight[highlight]
            )
        
        return [ball]
    
    def is_far(self,candidate):
        return self.location.is_far(candidate,self.radius+1)
    
    def __str__(self):
        return f"{self.color} ball at {self.location}"




  