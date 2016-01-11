local utils = require 'misc.utils'
local net_utils = {}

function net_utils.send_to_gpu(videos, labels, ids)
  if type(videos) == 'table' then
    for i=1,#videos do
      videos[i] = videos[i]:float():cuda()
    end
  else
    videos = videos:float():cuda()
  end

  if labels ~= nil then
    labels = labels:float():cuda()
  end

  return videos, labels, ids
end

function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end

function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end

function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

return net_utils