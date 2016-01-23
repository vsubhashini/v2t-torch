require 'nn'
local utils = require 'misc.utils'
local AttentionMLP = require 'misc.AttentionMLP'

local model, parent = torch.class('nn.AttentionModel', 'nn.Module')
function model:__init(opt)
  parent.__init(self)

  -- options for core network
  self.batch_size = utils.getopt(opt, 'batch_size') 
  self.num_annotations = utils.getopt(opt, 'num_annotations')
  self.annotation_size = utils.getopt(opt, 'annotation_size')

  -- create the core lstm network. note +1 for both the START and END tokens
  self.mlp = AttentionMLP.attmlp(self.num_annotations, self.annotation_size)
end

function model:updateOutput(input)
	local img = input:select(1, 1)
	print(img:size())
	local probs = self.mlp:forward(img)
	print(probs)
	print(probs:size())
end
-------------------------------------------------------------------------------
-- Context Extractor -- Generates input to Attention Model
-------------------------------------------------------------------------------
local extractor, parent = torch.class('nn.AnnotationExtractor', 'nn.Module')
function extractor:__init(size, stride, padding)
	parent.__init(self)

	self.size = size
	self.stride = stride
	self.padding = padding
end

function extractor:updateOutput(input)
	local batchSize = input:size(1)
	local filters = input:size(2)
	local height = input:size(3)
	local width = input:size(4)

	local height_fit = (height - self.size + 2*self.padding)/self.stride + 1
	local width_fit =  (width - self.size + 2*self.padding)/self.stride + 1

	assert(height_fit%1 == 0 and width_fit%1 == 0, "extractor parameters are invalid")

	-- self.output:resize(self.size, self.size, height_fit*width_fit)
	self.output:resize(batchSize, height_fit*width_fit, self.size*self.size*filters)

	local pad
	if self.padding > 0 then
		pad = torch.Tensor(height + 2*self.padding, width + 2*self.padding)
		pad:sub(self.padding, height-self.padding, self.padding, width-self.padding):copy(input)
	else
		pad = input
	end

	for h_c=1,height_fit do
		local h_ix = (h_c-1)*self.stride + 1
		for w_c=1,width_fit do
			local w_ix = (w_c-1)*self.stride + 1
			local batchFeats = input:sub(1,batchSize,1,filters,h_ix,h_ix+self.size-1,w_ix,w_ix+self.size-1):resize(batchSize, filters*self.size*self.size)
			self.output:select(2, (h_c-1)*width_fit + w_c):copy(batchFeats)
			-- self.output:select(1, (h_c-1)*width_fit + w_c):copy(input:sub())
			-- self.output:select(3, h_c*w_c):copy(pad:narrow(2, w_ix, self.size))
		end
	end

	return self.output
end