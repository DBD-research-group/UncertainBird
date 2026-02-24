<template>
    <div class="border-2 border-black rounded-lg bg-gray-300 grid grid-cols-12">
        <div class="col-span-1 flex items-center justify-center border-black p-2">
            <img src="/figures/icons/rq.drawio.svg" class="h-12 w-12 " />
        </div>
        <div class="col-span-11 p-4 flex flex-col justify-center">
            <p>
                <slot mdc-unwrap="p" />
            </p>
        </div>
    </div>
</template>